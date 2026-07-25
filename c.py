import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib import error, request

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    from obspy.core import Stream

    OBSPY_AVAILABLE = True
except ImportError:
    OBSPY_AVAILABLE = False
    print("[WARN] obspy not available")


FDSN_URLS = [
    os.getenv("FDSN_SERVER", "http://61.19.55.90:8787"),
    "http://61.19.55.90:8788",
    "http://61.19.55.90:80",
]
ACTIVE_FDSN_SERVER = None

# Wat Arun station: shown as TMP027 in the realtime UI, queried as TMP27.
STATION = os.getenv("STATION", "TMP24")
NETWORK = os.getenv("NETWORK", "MU")
CHANNELS = ["ENE", "ENN", "ENZ"]

FETCH_WINDOW = int(os.getenv("FETCH_WINDOW", "90"))
CACHE_INTERVAL = int(os.getenv("CACHE_INTERVAL", "5"))
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "500"))
API_VALUE_DIVISOR = float(os.getenv("API_VALUE_DIVISOR", "26164"))
EXCEEDANCE_THRESHOLD = float(os.getenv("EXCEEDANCE_THRESHOLD", "0.02"))

SAVE_EXCEEDANCES = os.getenv("SAVE_EXCEEDANCES", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "exceedance_events")

BANGKOK_TZ = timezone(timedelta(hours=7))


app = FastAPI(
    title="Earthquake Data API",
    description="Real-time Earthquake Waveform API",
    version="3.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fdsn_client = None
cached_data = None
last_fetch_time = None
is_fetching = False
saved_exceedance_keys = set()


class AccelerationData(BaseModel):
    x: List[float]
    y: List[float]
    z: List[float]


class ExceedanceRange(BaseModel):
    station: str
    network: str
    threshold: float
    start_index: int
    end_index: int
    start_time: str
    end_time: str
    peak_index: int
    peak_time: str
    peak_abs_value: float
    peak_x: float
    peak_y: float
    peak_z: float
    duration_seconds: float


class EarthquakeData(BaseModel):
    timestamp: str
    server_timestamp: str
    acceleration: AccelerationData
    intensity_peak: float
    exceedance_threshold: float
    exceedance_ranges: List[ExceedanceRange] = Field(default_factory=list)
    cached: bool = False


def thai_now() -> datetime:
    return datetime.now(timezone.utc).astimezone(BANGKOK_TZ)


def utcdate_to_bangkok_iso(value) -> str:
    dt = value.datetime.replace(tzinfo=timezone.utc)
    return dt.astimezone(BANGKOK_TZ).isoformat()


def model_dump(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def init_fdsn():
    global fdsn_client, ACTIVE_FDSN_SERVER
    if not OBSPY_AVAILABLE:
        return False

    for url in FDSN_URLS:
        try:
            temp_client = Client(url, timeout=15)
            temp_client.get_stations(
                network=NETWORK,
                station=STATION,
                level="station",
            )
            fdsn_client = temp_client
            ACTIVE_FDSN_SERVER = url
            print(f"[INFO] Connected to: {url}")
            return True
        except Exception:
            continue

    return False


def detect_exceedance_ranges(
    acc: Dict[str, List[float]],
    sample_start_time,
    sampling_rate: float,
    threshold: float,
) -> List[ExceedanceRange]:
    if not sample_start_time or sampling_rate <= 0:
        return []

    sample_count = min(len(acc["x"]), len(acc["y"]), len(acc["z"]))
    ranges: List[ExceedanceRange] = []
    active_start = None
    active_peak = None
    active_peak_value = 0.0

    def peak_abs(index: int) -> float:
        return max(abs(acc["x"][index]), abs(acc["y"][index]), abs(acc["z"][index]))

    def close_range(end_index: int):
        nonlocal active_start, active_peak, active_peak_value
        start_time = sample_start_time + (active_start / sampling_rate)
        end_time = sample_start_time + (end_index / sampling_rate)
        peak_time = sample_start_time + (active_peak / sampling_rate)
        ranges.append(
            ExceedanceRange(
                station=STATION,
                network=NETWORK,
                threshold=threshold,
                start_index=active_start,
                end_index=end_index,
                start_time=utcdate_to_bangkok_iso(start_time),
                end_time=utcdate_to_bangkok_iso(end_time),
                peak_index=active_peak,
                peak_time=utcdate_to_bangkok_iso(peak_time),
                peak_abs_value=active_peak_value,
                peak_x=acc["x"][active_peak],
                peak_y=acc["y"][active_peak],
                peak_z=acc["z"][active_peak],
                duration_seconds=((end_index - active_start) + 1) / sampling_rate,
            )
        )
        active_start = None
        active_peak = None
        active_peak_value = 0.0

    for index in range(sample_count):
        value = peak_abs(index)
        if value > threshold:
            if active_start is None:
                active_start = index
                active_peak = index
                active_peak_value = value
            elif value > active_peak_value:
                active_peak = index
                active_peak_value = value
        elif active_start is not None:
            close_range(index - 1)

    if active_start is not None:
        close_range(sample_count - 1)

    return ranges


def save_exceedances_to_supabase(ranges: List[ExceedanceRange]) -> None:
    if not SAVE_EXCEEDANCES or not ranges:
        return
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        print("[WARN] Supabase env vars not set; exceedance ranges were not saved")
        return

    rows = []
    for item in ranges:
        dedupe_key = (item.station, item.start_time, item.end_time)
        if dedupe_key in saved_exceedance_keys:
            continue
        rows.append(model_dump(item))
        saved_exceedance_keys.add(dedupe_key)

    if not rows:
        return

    url = (
        f"{SUPABASE_URL.rstrip('/')}/rest/v1/{SUPABASE_TABLE}"
        "?on_conflict=station,start_time,end_time"
    )
    req = request.Request(
        url,
        data=json.dumps(rows).encode("utf-8"),
        method="POST",
        headers={
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "resolution=ignore-duplicates,return=minimal",
        },
    )

    try:
        with request.urlopen(req, timeout=10) as response:
            if response.status not in (200, 201, 204):
                print(f"[WARN] Supabase insert returned HTTP {response.status}")
    except error.HTTPError as exc:
        print(f"[WARN] Supabase insert failed: HTTP {exc.code} {exc.reason}")
    except Exception as exc:
        print(f"[WARN] Supabase insert failed: {exc}")


def fetch_earthquake_data_sync() -> Optional[EarthquakeData]:
    global cached_data, last_fetch_time, is_fetching, fdsn_client
    if is_fetching:
        return cached_data
    is_fetching = True

    try:
        if not OBSPY_AVAILABLE:
            return cached_data
        if fdsn_client is None:
            if not init_fdsn():
                return cached_data

        end = UTCDateTime.now()
        start = end - FETCH_WINDOW
        stream = Stream()

        for ch in CHANNELS:
            try:
                trace = fdsn_client.get_waveforms(NETWORK, STATION, "00", ch, start, end)
                stream += trace
            except Exception:
                pass

        if len(stream) == 0:
            return cached_data

        try:
            stream.merge(method=0, fill_value="interpolate", interpolation_samples=0)
        except Exception:
            return cached_data

        acc = {"x": [], "y": [], "z": []}
        sample_meta: Dict[str, Dict[str, Any]] = {}

        for trace in stream:
            if len(trace.data) == 0:
                continue

            window_samples = [float(val) / API_VALUE_DIVISOR for val in trace.data]
            window_mean = sum(window_samples) / len(window_samples)
            recent_samples = window_samples[-min(NUM_SAMPLES, len(window_samples)) :]
            samples_list = [val - window_mean for val in recent_samples]
            sample_start_index = len(trace.data) - len(recent_samples)
            meta = {
                "start_time": trace.stats.starttime
                + (sample_start_index / trace.stats.sampling_rate),
                "sampling_rate": float(trace.stats.sampling_rate),
            }

            ch_name = trace.stats.channel
            if ch_name == "ENE":
                acc["x"] = samples_list
                sample_meta["x"] = meta
            elif ch_name == "ENN":
                acc["y"] = samples_list
                sample_meta["y"] = meta
            elif ch_name == "ENZ":
                acc["z"] = samples_list
                sample_meta["z"] = meta

        for axis in ["x", "y", "z"]:
            if not acc[axis]:
                acc[axis] = [0.0] * NUM_SAMPLES

        peak_x = max(abs(v) for v in acc["x"])
        peak_y = max(abs(v) for v in acc["y"])
        peak_z = max(abs(v) for v in acc["z"])
        intensity = max(peak_x, peak_y, peak_z)

        ref_meta = sample_meta.get("z") or sample_meta.get("x") or sample_meta.get("y")
        exceedance_ranges: List[ExceedanceRange] = []
        if ref_meta:
            exceedance_ranges = detect_exceedance_ranges(
                acc=acc,
                sample_start_time=ref_meta["start_time"],
                sampling_rate=ref_meta["sampling_rate"],
                threshold=EXCEEDANCE_THRESHOLD,
            )
            save_exceedances_to_supabase(exceedance_ranges)

        thai_time_now = thai_now()
        data_time = thai_time_now - timedelta(seconds=5)

        data = EarthquakeData(
            timestamp=data_time.isoformat(),
            server_timestamp=thai_time_now.isoformat(),
            acceleration=AccelerationData(**acc),
            intensity_peak=intensity,
            exceedance_threshold=EXCEEDANCE_THRESHOLD,
            exceedance_ranges=exceedance_ranges,
            cached=False,
        )
        cached_data = data
        last_fetch_time = thai_time_now
        return data

    except Exception as e:
        print(f"[ERROR] {e}")
        fdsn_client = None
        return cached_data
    finally:
        is_fetching = False


async def background_fetch_loop():
    while True:
        try:
            await asyncio.get_event_loop().run_in_executor(None, fetch_earthquake_data_sync)
        except Exception:
            pass
        await asyncio.sleep(CACHE_INTERVAL)


@app.on_event("startup")
async def startup():
    init_fdsn()
    asyncio.create_task(background_fetch_loop())


@app.get("/data", response_model=EarthquakeData, tags=["Data"])
async def get_data():
    current_thai_time = thai_now()
    if cached_data:
        return EarthquakeData(
            timestamp=cached_data.timestamp,
            server_timestamp=current_thai_time.isoformat(),
            acceleration=cached_data.acceleration,
            intensity_peak=cached_data.intensity_peak,
            exceedance_threshold=cached_data.exceedance_threshold,
            exceedance_ranges=cached_data.exceedance_ranges,
            cached=True,
        )

    empty_arr = [0.0] * NUM_SAMPLES
    return EarthquakeData(
        timestamp=(current_thai_time - timedelta(seconds=5)).isoformat(),
        server_timestamp=current_thai_time.isoformat(),
        acceleration=AccelerationData(x=empty_arr, y=empty_arr, z=empty_arr),
        intensity_peak=0,
        exceedance_threshold=EXCEEDANCE_THRESHOLD,
        exceedance_ranges=[],
        cached=False,
    )


@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "obspy_available": OBSPY_AVAILABLE,
        "active_fdsn_server": ACTIVE_FDSN_SERVER,
        "station": STATION,
        "network": NETWORK,
        "exceedance_threshold": EXCEEDANCE_THRESHOLD,
        "value_divisor": API_VALUE_DIVISOR,
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
