create table if not exists public.exceedance_events (
  id uuid primary key default gen_random_uuid(),
  station text not null,
  network text not null,
  threshold_mg double precision not null,
  start_index integer not null,
  end_index integer not null,
  start_time timestamptz not null,
  end_time timestamptz not null,
  peak_index integer not null,
  peak_time timestamptz not null,
  peak_mg double precision not null,
  peak_x_mg double precision not null,
  peak_y_mg double precision not null,
  peak_z_mg double precision not null,
  duration_seconds double precision not null,
  created_at timestamptz not null default now(),
  unique (station, start_time, end_time)
);

create index if not exists exceedance_events_start_time_idx
  on public.exceedance_events (start_time desc);
