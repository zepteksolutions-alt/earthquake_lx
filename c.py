import asyncio
import json
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib import error, request

import numpy as np
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

STATION = os.getenv("STATION", "TMP24")
NETWORK = os.getenv("NETWORK", "MU")
CHANNELS = ["ENE", "ENN", "ENZ"]

FETCH_WINDOW = int(os.getenv("FETCH_WINDOW", "90"))
CACHE_INTERVAL = int(os.getenv("CACHE_INTERVAL", "5"))
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "500"))
G_CONST = 9.80665
MG_TO_MPS2 = 0.00981
SENSITIVITY_COUNTS_PER_MPS2 = float(os.getenv("SENSITIVITY_COUNTS_PER_MPS2", "26164"))
PGA_THRESHOLD_MPS2 = float(os.getenv("PGA_THRESHOLD_MPS2", "0.02"))
THRESHOLD_MG = float(os.getenv("THRESHOLD_MG", str(PGA_THRESHOLD_MPS2 / MG_TO_MPS2)))

SAVE_EXCEEDANCES = os.getenv("SAVE_EXCEEDANCES", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "exceedance_events")

EVENT_HANGOVER_SECONDS = float(os.getenv("EVENT_HANGOVER_SECONDS", "3.0"))
EVENT_MIN_DURATION_SECONDS = float(os.getenv("EVENT_MIN_DURATION_SECONDS", "0.3"))

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
station_inventory = None
cached_data = None
last_fetch_time = None
is_fetching = False

# Event-detection state — survives across fetch cycles, reset on Railway restart.
# On restart we re-scan the last 90 s window; Supabase's on_conflict policy
# de-duplicates any event that was already saved before the restart.
event_last_processed_time = None  # UTCDateTime; samples older than this are skipped
active_event = None  # dict: { start_time, peak_time, peak_mg, peak_x_mg, peak_y_mg, peak_z_mg, last_above_time, cooldown_seconds }


class AccelerationData(BaseModel):
    x: List[float]
    y: List[float]
    z: List[float]


class ExceedanceRange(BaseModel):
    station: str
    network: str
    threshold_mg: float
    start_index: int
    end_index: int
    start_time: str
    end_time: str
    peak_index: int
    peak_time: str
    peak_mg: float
    peak_x_mg: float
    peak_y_mg: float
    peak_z_mg: float
    duration_seconds: float


class EarthquakeData(BaseModel):
    timestamp: str
    server_timestamp: str
    acceleration: AccelerationData
    acceleration_mg: AccelerationData
    intensity_peak: float
    intensity_peak_mg: float
    threshold_mg: float
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
    global fdsn_client, station_inventory, ACTIVE_FDSN_SERVER
    if not OBSPY_AVAILABLE:
        return False

    for url in FDSN_URLS:
        try:
            temp_client = Client(url, timeout=15)
            station_inventory = temp_client.get_stations(
                network=NETWORK,
                station=STATION,
                level="RESP",
            )
            fdsn_client = temp_client
            ACTIVE_FDSN_SERVER = url
            print(f"[INFO] Connected to: {url}")
            return True
        except Exception:
            continue

    return False


def update_event_state(
    acc_mg: Dict[str, List[float]],
    sample_start_time,
    sampling_rate: float,
    pga_threshold_mps2: float,
    threshold_mg: float,
) -> List[ExceedanceRange]:
    """
    Stateful event detector. Returns a list of FINALIZED events (zero or one per call,
    occasionally more if multiple events end within one window).

    State machine:
      IDLE     → vector > threshold     → ACTIVE (record start_time, peak)
      ACTIVE   → vector > threshold     → update peak if greater; reset cooldown
      ACTIVE   → vector <= threshold    → accumulate cooldown_seconds
      cooldown ≥ EVENT_HANGOVER_SECONDS → finalize event, return it, → IDLE
    """
    global event_last_processed_time, active_event

    if not sample_start_time or sampling_rate <= 0:
        return []

    sample_count = min(len(acc_mg["x"]), len(acc_mg["y"]), len(acc_mg["z"]))
    if sample_count == 0:
        return []

    sample_dt = 1.0 / sampling_rate
    finalized: List[ExceedanceRange] = []

    def vector_mg_at(index: int) -> float:
        return math.sqrt(
            acc_mg["x"][index] ** 2
            + acc_mg["y"][index] ** 2
            + acc_mg["z"][index] ** 2
        )

    def finalize_active():
        nonlocal finalized
        global active_event
        if active_event is None:
            return
        duration = float(active_event["last_above_time"] - active_event["start_time"])
        if duration < EVENT_MIN_DURATION_SECONDS:
            # discard spike too short to count as an event
            active_event = None
            return
        finalized.append(
            ExceedanceRange(
                station=STATION,
                network=NETWORK,
                threshold_mg=threshold_mg,
                start_index=0,
                end_index=0,
                start_time=utcdate_to_bangkok_iso(active_event["start_time"]),
                end_time=utcdate_to_bangkok_iso(active_event["last_above_time"]),
                peak_index=0,
                peak_time=utcdate_to_bangkok_iso(active_event["peak_time"]),
                peak_mg=active_event["peak_mg"],
                peak_x_mg=active_event["peak_x_mg"],
                peak_y_mg=active_event["peak_y_mg"],
                peak_z_mg=active_event["peak_z_mg"],
                duration_seconds=max(duration, 0.0),
            )
        )
        active_event = None

    for index in range(sample_count):
        sample_time = sample_start_time + (index * sample_dt)
        # Skip samples already processed in a previous fetch (overlap protection)
        if event_last_processed_time is not None and sample_time <= event_last_processed_time:
            continue

        value_mg = vector_mg_at(index)
        value_mps2 = value_mg * MG_TO_MPS2

        if value_mps2 > pga_threshold_mps2:
            if active_event is None:
                active_event = {
                    "start_time": sample_time,
                    "peak_time": sample_time,
                    "peak_mg": value_mg,
                    "peak_x_mg": acc_mg["x"][index],
                    "peak_y_mg": acc_mg["y"][index],
                    "peak_z_mg": acc_mg["z"][index],
                    "last_above_time": sample_time,
                    "cooldown_seconds": 0.0,
                }
            else:
                active_event["last_above_time"] = sample_time
                active_event["cooldown_seconds"] = 0.0
                if value_mg > active_event["peak_mg"]:
                    active_event["peak_mg"] = value_mg
                    active_event["peak_time"] = sample_time
                    active_event["peak_x_mg"] = acc_mg["x"][index]
                    active_event["peak_y_mg"] = acc_mg["y"][index]
                    active_event["peak_z_mg"] = acc_mg["z"][index]
        else:
            if active_event is not None:
                active_event["cooldown_seconds"] += sample_dt
                if active_event["cooldown_seconds"] >= EVENT_HANGOVER_SECONDS:
                    finalize_active()

        event_last_processed_time = sample_time

    return finalized


def save_exceedances_to_supabase(events: List[ExceedanceRange]) -> None:
    if not SAVE_EXCEEDANCES or not events:
        return
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        print("[WARN] Supabase env vars not set; exceedance events were not saved")
        return

    rows = [model_dump(item) for item in events]

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
    global cached_data, last_fetch_time, is_fetching, station_inventory, fdsn_client
    if is_fetching:
        return cached_data
    is_fetching = True

    try:
        if not OBSPY_AVAILABLE:
            return cached_data
        if fdsn_client is None or station_inventory is None:
            if not init_fdsn():
                return cached_data

        end = UTCDateTime.now()
        start = end - FETCH_WINDOW
        stream = Stream()
        failed_channels: List[str] = []

        for ch in CHANNELS:
            try:
                trace = fdsn_client.get_waveforms(NETWORK, STATION, "00", ch, start, end)
                stream += trace
            except Exception as exc:
                failed_channels.append(ch)
                print(f"[WARN] fetch {ch} failed: {exc}")

        # If any channel failed, keep last-good cache instead of injecting zeros.
        # A partial waveform with one axis flat at 0 misleads peak detection and exceedance state.
        if failed_channels:
            print(f"[WARN] incomplete fetch: missing {failed_channels} — keeping cached data")
            return cached_data

        try:
            stream.merge(method=0, fill_value="interpolate", interpolation_samples=0)
        except Exception as exc:
            print(f"[WARN] stream.merge failed: {exc}")
            return cached_data

        channels_present = {t.stats.channel for t in stream}
        if not {"ENE", "ENN", "ENZ"}.issubset(channels_present):
            print(f"[WARN] channels after merge: {channels_present} — keeping cached data")
            return cached_data

        # Align all axes to a shared time window before slicing the trailing samples.
        # Without this, each trace independently took its last 500 samples and the three
        # graphs ended up at slightly different wall-clock times.
        try:
            common_start = max(t.stats.starttime for t in stream)
            common_end = min(t.stats.endtime for t in stream)
            if common_end - common_start <= 0:
                print("[WARN] no overlapping time window across channels — keeping cached data")
                return cached_data
            stream.trim(starttime=common_start, endtime=common_end)
        except Exception as exc:
            print(f"[WARN] trim failed: {exc}")
            return cached_data

        # Reject masked traces (gap in the trimmed window) rather than letting NaN-ish
        # values silently become 0 downstream.
        for t in stream:
            data = t.data
            if hasattr(data, "mask") and np.ma.is_masked(data) and bool(np.any(data.mask)):
                print(f"[WARN] {t.stats.channel}: gap in trimmed window — keeping cached data")
                return cached_data

        try:
            stream.detrend("demean")
            for trace in stream:
                trace.data = trace.data / SENSITIVITY_COUNTS_PER_MPS2
        except Exception as exc:
            print(f"[WARN] detrend/scale failed: {exc}")
            return cached_data

        acc = {"x": [], "y": [], "z": []}
        sample_meta: Dict[str, Dict[str, Any]] = {}

        for trace in stream:
            if len(trace.data) == 0:
                continue

            recent_samples = trace.data[-min(NUM_SAMPLES, len(trace.data)) :]
            samples_list = [float(val) for val in recent_samples]
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

        # All three axes must be non-empty — abort and keep cache otherwise.
        if not (acc["x"] and acc["y"] and acc["z"]):
            print("[WARN] one or more axes empty after slicing — keeping cached data")
            return cached_data

        acc_mg = {
            "x": [(v / G_CONST) * 1000 for v in acc["x"]],
            "y": [(v / G_CONST) * 1000 for v in acc["y"]],
            "z": [(v / G_CONST) * 1000 for v in acc["z"]],
        }

        # PGA = max over t of |a(t)|, where a is the 3D vector. Peak per axis taken
        # at different times (the old method) inflates the magnitude.
        ax = np.asarray(acc["x"], dtype=float)
        ay = np.asarray(acc["y"], dtype=float)
        az = np.asarray(acc["z"], dtype=float)
        vector_at_t = np.sqrt(ax * ax + ay * ay + az * az)
        intensity = float(vector_at_t.max()) if vector_at_t.size > 0 else 0.0

        ref_meta = sample_meta.get("z") or sample_meta.get("x") or sample_meta.get("y")
        exceedance_events: List[ExceedanceRange] = []
        if ref_meta:
            exceedance_events = update_event_state(
                acc_mg=acc_mg,
                sample_start_time=ref_meta["start_time"],
                sampling_rate=ref_meta["sampling_rate"],
                pga_threshold_mps2=PGA_THRESHOLD_MPS2,
                threshold_mg=THRESHOLD_MG,
            )
            save_exceedances_to_supabase(exceedance_events)

        thai_time_now = thai_now()
        data_time = thai_time_now - timedelta(seconds=5)

        data = EarthquakeData(
            timestamp=data_time.isoformat(),
            server_timestamp=thai_time_now.isoformat(),
            acceleration=AccelerationData(**acc),
            acceleration_mg=AccelerationData(**acc_mg),
            intensity_peak=intensity,
            intensity_peak_mg=intensity / G_CONST * 1000,
            threshold_mg=THRESHOLD_MG,
            exceedance_ranges=exceedance_events,
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
            acceleration_mg=cached_data.acceleration_mg,
            intensity_peak=cached_data.intensity_peak,
            intensity_peak_mg=cached_data.intensity_peak_mg,
            threshold_mg=cached_data.threshold_mg,
            exceedance_ranges=cached_data.exceedance_ranges,
            cached=True,
        )

    empty_arr = [0.0] * NUM_SAMPLES
    return EarthquakeData(
        timestamp=(current_thai_time - timedelta(seconds=5)).isoformat(),
        server_timestamp=current_thai_time.isoformat(),
        acceleration=AccelerationData(x=empty_arr, y=empty_arr, z=empty_arr),
        acceleration_mg=AccelerationData(x=empty_arr, y=empty_arr, z=empty_arr),
        intensity_peak=0,
        intensity_peak_mg=0,
        threshold_mg=THRESHOLD_MG,
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
        "sensitivity_counts_per_mps2": SENSITIVITY_COUNTS_PER_MPS2,
        "pga_threshold_mps2": PGA_THRESHOLD_MPS2,
        "threshold_mg": THRESHOLD_MG,
        "event_hangover_seconds": EVENT_HANGOVER_SECONDS,
        "event_min_duration_seconds": EVENT_MIN_DURATION_SECONDS,
        "active_event_in_progress": active_event is not None,
        "last_fetch_time": last_fetch_time.isoformat() if last_fetch_time else None,
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
