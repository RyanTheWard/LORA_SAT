# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.25.2",
#     "matplotlib==3.11.2",
#     "numpy==2.4.6",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    # Physical & orbital constants
    C_LIGHT = 299_792_458.0       # Speed of light (m/s)
    MU_EARTH = 3.986004418e14     # Earth gravitational parameter (m^3/s^2)
    R_EARTH_M = 6_371_000.0       # Mean Earth radius (m)
    R_EARTH_KM = 6_371.0          # Mean Earth radius (km)
    OMEGA_EARTH = 7.2921159e-5    # Earth rotation rate (rad/s)

    # Standard LoRa parameter search space
    LORA_BWS_KHZ = (62.5, 125.0, 250.0, 500.0)
    LORA_SFS = (7, 8, 9, 10, 11, 12)
    LORA_CRS = {
        "4/5 (CR=1)": 4.0 / 5.0,
        "4/6 (CR=2)": 4.0 / 6.0,
        "4/7 (CR=3)": 4.0 / 7.0,
        "4/8 (CR=4)": 4.0 / 8.0,
    }
    LORA_SNR_THRESHOLDS_DB = {
        7: -7.5,
        8: -10.0,
        9: -12.5,
        10: -15.0,
        11: -17.5,
        12: -20.0,
    }

    return (
        AutoMinorLocator,
        C_LIGHT,
        LORA_BWS_KHZ,
        LORA_CRS,
        LORA_SFS,
        LORA_SNR_THRESHOLDS_DB,
        MU_EARTH,
        OMEGA_EARTH,
        R_EARTH_KM,
        R_EARTH_M,
        mo,
        np,
        plt,
    )


@app.cell
def _(
    C_LIGHT,
    LORA_SNR_THRESHOLDS_DB,
    MU_EARTH,
    OMEGA_EARTH,
    R_EARTH_KM,
    R_EARTH_M,
    np,
):
    def compute_ergodic_contact_curve(
        altitude_km,
        inclination_deg,
        station_lat_deg,
        ranges_km,
        min_elevation_deg=0.0,
        n_quad=512,
    ):
        """
        Analytical ergodic (T -> infinity) satellite-ground station contact
        statistics versus slant range for a circular orbit on a spherical Earth.
        """
        mu_km = 398600.4418
        r_orbit = R_EARTH_KM + altitude_km
        n_mean_motion = np.sqrt(mu_km / r_orbit**3)

        min_el_rad = np.radians(min_elevation_deg)
        psi_horizon = (
            np.arccos(np.clip((R_EARTH_KM * np.cos(min_el_rad)) / r_orbit, -1.0, 1.0))
            - min_el_rad
        )
        cos_psi_horizon = np.cos(psi_horizon)

        cos_psi_range = (R_EARTH_KM**2 + r_orbit**2 - ranges_km**2) / (
            2.0 * R_EARTH_KM * r_orbit
        )
        cos_psi_eff = np.clip(np.maximum(cos_psi_range, cos_psi_horizon), -1.0, 1.0)
        cos_psi_eff = np.where(ranges_km < altitude_km, 1.0, cos_psi_eff)

        inc_rad = np.radians(inclination_deg)
        lat_gs_rad = np.radians(station_lat_deg)
        sin_lat_gs = np.sin(lat_gs_rad)
        cos_lat_gs = np.cos(lat_gs_rad)

        u_grid = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_quad)
        du = u_grid[1] - u_grid[0]

        sin_phi = np.sin(inc_rad) * np.sin(u_grid)
        cos_phi = np.sqrt(np.maximum(1.0 - sin_phi**2, 1e-15))

        denom = np.maximum(cos_lat_gs * cos_phi[:, None], 1e-15)
        cos_dlambda = (cos_psi_eff[None, :] - sin_lat_gs * sin_phi[:, None]) / denom

        delta_lambda = np.where(
            cos_psi_eff[None, :] < 1.0,
            np.arccos(np.clip(cos_dlambda, -1.0, 1.0)),
            0.0,
        )

        contact_fraction = np.trapezoid(delta_lambda, dx=du, axis=0) / (np.pi**2)
        contact_fraction = np.where(ranges_km < altitude_km, 0.0, contact_fraction)
        avg_contact_min_per_day = 1440.0 * contact_fraction

        lambda_dot_rel = np.abs(
            (n_mean_motion * np.cos(inc_rad)) / (cos_phi**2) - OMEGA_EARTH
        )[:, None]
        n_d_delta_lambda_du = n_mean_motion * np.abs(np.gradient(delta_lambda, du, axis=0))

        boundary_flux = np.where(
            delta_lambda > 1e-9,
            np.maximum(lambda_dot_rel, n_d_delta_lambda_du),
            0.0,
        )
        contacts_per_sec = np.trapezoid(boundary_flux, dx=du, axis=0) / (2.0 * (np.pi**2))
        avg_contacts_per_day = np.where(ranges_km < altitude_km, 0.0, 86400.0 * contacts_per_sec)

        return avg_contact_min_per_day, avg_contacts_per_day

    def compute_pass_geometry(
        orbit_alt_km,
        orbit_inc_deg,
        gs_lat_deg,
        power_tx_w,
        freq_mhz,
        gt_dbi,
        gr_dbi,
    ):
        """
        Compute all fixed orbital, Doppler, Friis received power, and ergodic
        contact profiles across a pass from zenith (d = h) to radar horizon.
        """
        freq_hz = freq_mhz * 1e6
        wavelength = C_LIGHT / freq_hz
        gt_linear = 10.0 ** (gt_dbi / 10.0)
        gr_linear = 10.0 ** (gr_dbi / 10.0)
        eirp_dbm = 10.0 * np.log10(power_tx_w * 1000.0) + gt_dbi

        h_m = orbit_alt_km * 1000.0
        r_sat_m = R_EARTH_M + h_m
        v_s = np.sqrt(MU_EARTH / r_sat_m)

        horizon_km = np.sqrt(r_sat_m**2 - R_EARTH_M**2) / 1000.0
        ground_horizon_km = (R_EARTH_M * np.arccos(R_EARTH_M / r_sat_m)) / 1000.0
        theta_horizon_rad = np.arccos(R_EARTH_M / r_sat_m)

        range_km = np.linspace(orbit_alt_km, horizon_km, 500)
        range_m = range_km * 1000.0

        pr_w = power_tx_w * gt_linear * gr_linear * ((wavelength / (4.0 * np.pi * range_m)) ** 2)
        pr_dbm = 10.0 * np.log10(pr_w * 1000.0)
        fspl_db = 20.0 * np.log10((4.0 * np.pi * range_m) / wavelength)

        cos_theta = np.clip(
            (R_EARTH_M**2 + r_sat_m**2 - range_m**2) / (2.0 * R_EARTH_M * r_sat_m),
            -1.0,
            1.0,
        )
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        v_los = v_s * R_EARTH_M * sin_theta / range_m

        doppler_hz = freq_hz * (v_los / C_LIGHT)
        doppler_shift_max_hz = float(doppler_hz[-1])

        radial_accel = ((v_s**2) * (R_EARTH_M / r_sat_m) * cos_theta - v_los**2) / range_m
        doppler_rate_hz_s = -freq_hz * (radial_accel / C_LIGHT)
        doppler_rate_mag_hz_s = np.abs(doppler_rate_hz_s)
        max_doppler_rate_hz_s = float(doppler_rate_mag_hz_s[0])

        erg_ranges_km = np.linspace(0.0, horizon_km * 1.15, 160)
        erg_min_per_day, erg_contacts_per_day = compute_ergodic_contact_curve(
            altitude_km=orbit_alt_km,
            inclination_deg=orbit_inc_deg,
            station_lat_deg=gs_lat_deg,
            ranges_km=erg_ranges_km,
        )
        max_ergodic_min_per_day = float(
            np.interp(horizon_km, erg_ranges_km, erg_min_per_day)
        )

        return {
            "alt_km": orbit_alt_km,
            "inc_deg": orbit_inc_deg,
            "gs_lat_deg": gs_lat_deg,
            "horizon_km": horizon_km,
            "ground_horizon_km": ground_horizon_km,
            "theta_horizon_rad": theta_horizon_rad,
            "v_s": v_s,
            "freq_hz": freq_hz,
            "wavelength": wavelength,
            "eirp_dbm": eirp_dbm,
            "range_km": range_km,
            "pr_dbm": pr_dbm,
            "fspl_db": fspl_db,
            "doppler_hz": doppler_hz,
            "doppler_shift_max_hz": doppler_shift_max_hz,
            "doppler_rate_hz_s": doppler_rate_hz_s,
            "doppler_rate_mag_hz_s": doppler_rate_mag_hz_s,
            "max_doppler_rate_hz_s": max_doppler_rate_hz_s,
            "erg_ranges_km": erg_ranges_km,
            "erg_min_per_day": erg_min_per_day,
            "erg_contacts_per_day": erg_contacts_per_day,
            "max_ergodic_min_per_day": max_ergodic_min_per_day,
        }

    def evaluate_lora_mode(
        bw_khz,
        sf,
        cr_label,
        cr_val,
        payload_bytes,
        rx_nf_db,
        req_fade_margin_db,
        doppler_mode,
        pass_geom,
    ):
        """
        Evaluate a single (BW, SF, CR) LoRa mode across the overhead pass geometry
        and return its link metrics, usable slant-range window [d_min, d_max],
        pass coverage %, and daily data yield.
        """
        bw_hz = bw_khz * 1e3
        t_sym_s = (2**sf) / bw_hz
        t_sym_ms = t_sym_s * 1000.0
        lora_bin_hz = bw_hz / (2**sf)
        ldro_enabled = t_sym_ms >= 16.0

        # Packet time-on-air (explicit header, CRC enabled, 8-symbol preamble)
        de = 1 if ldro_enabled else 0
        cr_denom = round(4.0 / cr_val)
        n_payload_sym = 8 + max(
            int(
                np.ceil(
                    (8 * payload_bytes - 4 * sf + 28 + 16) / (4 * (sf - 2 * de))
                )
            )
            * cr_denom,
            0,
        )
        t_packet_s = (8.0 + 4.25 + n_payload_sym) * t_sym_s

        nominal_bps = sf * (bw_hz / (2**sf)) * cr_val
        goodput_bps = (payload_bytes * 8.0) / t_packet_s

        snr_limit_db = LORA_SNR_THRESHOLDS_DB[sf]
        thermal_noise_dbm = -174.0 + 10.0 * np.log10(bw_hz) + rx_nf_db
        lora_sens_dbm = thermal_noise_dbm + snr_limit_db

        # Gate 1: Link margin across slant range
        margin_profile_db = pass_geom["pr_dbm"] - lora_sens_dbm
        zenith_margin_db = float(margin_profile_db[0])
        horizon_margin_db = float(margin_profile_db[-1])

        # Gate 2: Static Doppler limit
        if doppler_mode == "half_bw":
            doppler_limit_hz = 0.5 * bw_hz
        elif doppler_mode == "full_bw":
            doppler_limit_hz = 1.0 * bw_hz
        else:
            doppler_limit_hz = np.inf

        # Gate 3: Doppler rate (frequency drift) limits
        drift_margin_hz = (2.0 if ldro_enabled else 0.5) * lora_bin_hz
        pkt_drift_limit_hz_s = drift_margin_hz / t_packet_s
        chirp_slope_hz_s = lora_bin_hz / t_sym_s
        sym_chirp_limit_hz_s = 0.10 * chirp_slope_hz_s
        drift_limit_hz_s = min(pkt_drift_limit_hz_s, sym_chirp_limit_hz_s)

        # Pointwise pass feasibility mask
        snr_ok = margin_profile_db >= req_fade_margin_db
        doppler_ok = pass_geom["doppler_hz"] <= doppler_limit_hz
        drift_ok = pass_geom["doppler_rate_mag_hz_s"] <= drift_limit_hz_s
        usable_mask = snr_ok & doppler_ok & drift_ok

        d_grid = pass_geom["range_km"]
        alt_km = pass_geom["alt_km"]

        if np.any(usable_mask):
            usable_indices = np.where(usable_mask)[0]
            d_min_km = float(d_grid[usable_indices[0]])
            d_max_km = float(d_grid[usable_indices[-1]])
            pass_coverage_pct = float(np.mean(usable_mask) * 100.0)
            usable_min_per_day = float(
                max(
                    0.0,
                    np.interp(d_max_km, pass_geom["erg_ranges_km"], pass_geom["erg_min_per_day"])
                    - np.interp(d_min_km, pass_geom["erg_ranges_km"], pass_geom["erg_min_per_day"]),
                )
            )
        else:
            d_min_km = alt_km
            d_max_km = alt_km
            pass_coverage_pct = 0.0
            usable_min_per_day = 0.0

        daily_kb = (usable_min_per_day * 60.0 * goodput_bps) / (8.0 * 1024.0)
        daily_packets = (usable_min_per_day * 60.0) / t_packet_s

        reasons = []
        if not np.all(snr_ok):
            reasons.append("SNR Margin")
        if not np.all(doppler_ok):
            reasons.append("Static Doppler")
        if not np.all(drift_ok):
            reasons.append("Zenith Drift")
        bottleneck = "None (Full Pass)" if not reasons else " + ".join(reasons)

        drift_safety_ratio = drift_limit_hz_s / max(pass_geom["max_doppler_rate_hz_s"], 1e-9)

        return {
            "bw_khz": bw_khz,
            "bw_hz": bw_hz,
            "sf": sf,
            "cr_label": cr_label,
            "cr_val": cr_val,
            "t_sym_ms": t_sym_ms,
            "t_packet_s": t_packet_s,
            "ldro_enabled": ldro_enabled,
            "lora_bin_hz": lora_bin_hz,
            "nominal_bps": nominal_bps,
            "goodput_bps": goodput_bps,
            "snr_limit_db": snr_limit_db,
            "thermal_noise_dbm": thermal_noise_dbm,
            "lora_sens_dbm": lora_sens_dbm,
            "zenith_margin_db": zenith_margin_db,
            "horizon_margin_db": horizon_margin_db,
            "doppler_limit_hz": doppler_limit_hz,
            "drift_margin_hz": drift_margin_hz,
            "pkt_drift_limit_hz_s": pkt_drift_limit_hz_s,
            "sym_chirp_limit_hz_s": sym_chirp_limit_hz_s,
            "chirp_slope_hz_s": chirp_slope_hz_s,
            "drift_limit_hz_s": drift_limit_hz_s,
            "drift_safety_ratio": drift_safety_ratio,
            "d_min_km": d_min_km,
            "d_max_km": d_max_km,
            "zenith_blind_spot_km": max(0.0, d_min_km - alt_km),
            "pass_coverage_pct": pass_coverage_pct,
            "usable_min_per_day": usable_min_per_day,
            "daily_kb": daily_kb,
            "daily_packets": daily_packets,
            "bottleneck": bottleneck,
            "full_pass_closed": bool(np.all(usable_mask)),
        }

    return compute_ergodic_contact_curve, compute_pass_geometry, evaluate_lora_mode


@app.cell
def _(mo):
    # --- Section 1 UI: Mission Givens (Orbit, Ground Station & RF Hardware) ---
    alt_slider = mo.ui.slider(
        start=200,
        stop=2000,
        step=25,
        value=550,
        label="Orbital Altitude $h$ (km)",
        show_value=True,
    )
    inc_slider = mo.ui.slider(
        start=0.0,
        stop=120.0,
        step=0.5,
        value=51.6,
        label="Orbital Inclination $i$ (°)",
        show_value=True,
    )
    gs_lat_slider = mo.ui.slider(
        start=-70.0,
        stop=70.0,
        step=0.5,
        value=51.0,
        label="Ground Station Latitude $\\phi_{GS}$ (°)",
        show_value=True,
    )

    pt_input = mo.ui.number(
        value=0.1,
        start=0.01,
        stop=5.0,
        step=0.01,
        label="CubeSat Tx Power $P_t$ (W)",
    )
    freq_input = mo.ui.number(
        value=433.0,
        start=400.0,
        stop=450.0,
        step=0.5,
        label="Downlink Frequency $f$ (MHz)",
    )
    gt_input = mo.ui.number(
        value=2.15,
        start=-5.0,
        stop=12.0,
        step=0.25,
        label="CubeSat Tx Gain $G_t$ (dBi)",
    )
    gr_input = mo.ui.number(
        value=14.0,
        start=0.0,
        stop=25.0,
        step=0.5,
        label="Ground Station Rx Gain $G_r$ (dBi)",
    )
    nf_input = mo.ui.number(
        value=6.0,
        start=1.0,
        stop=15.0,
        step=0.5,
        label="Receiver Noise Figure $NF$ (dB)",
    )
    return (
        alt_slider,
        freq_input,
        gr_input,
        gs_lat_slider,
        gt_input,
        inc_slider,
        nf_input,
        pt_input,
    )


@app.cell
def _(
    alt_slider,
    compute_pass_geometry,
    freq_input,
    gr_input,
    gs_lat_slider,
    gt_input,
    inc_slider,
    mo,
    nf_input,
    pt_input,
):
    rx_nf_db = float(nf_input.value)
    pass_geom = compute_pass_geometry(
        orbit_alt_km=float(alt_slider.value),
        orbit_inc_deg=float(inc_slider.value),
        gs_lat_deg=float(gs_lat_slider.value),
        power_tx_w=float(pt_input.value),
        freq_mhz=float(freq_input.value),
        gt_dbi=float(gt_input.value),
        gr_dbi=float(gr_input.value),
    )

    mo.vstack(
        [
            mo.md(
                f"""
                ## 1. Mission Givens (Orbit & RF Hardware)

                Set the fixed orbital geometry and radio hardware parameters for the CubeSat downlink:

                {mo.hstack([alt_slider, inc_slider, gs_lat_slider], justify="start", wrap=True)}
                {mo.hstack([pt_input, freq_input, gt_input, gr_input, nf_input], justify="start", wrap=True)}
                """
            ),
            mo.hstack(
                [
                    mo.stat(
                        label="Zenith → Horizon Range",
                        value=f"{pass_geom['alt_km']:.0f} → {pass_geom['horizon_km']:.0f} km",
                        caption=f"Footprint arc: {pass_geom['ground_horizon_km']:.0f} km (v = {pass_geom['v_s']:.0f} m/s)",
                    ),
                    mo.stat(
                        label="Received Power (Zenith → Horizon)",
                        value=f"{pass_geom['pr_dbm'][0]:.1f} → {pass_geom['pr_dbm'][-1]:.1f} dBm",
                        caption=f"EIRP = {pass_geom['eirp_dbm']:.1f} dBm, FSPL = {pass_geom['fspl_db'][0]:.1f}–{pass_geom['fspl_db'][-1]:.1f} dB",
                    ),
                    mo.stat(
                        label="Max Horizon Doppler",
                        value=f"±{pass_geom['doppler_shift_max_hz'] / 1e3:.2f} kHz",
                        caption=f"Peak zenith drift: ±{pass_geom['max_doppler_rate_hz_s']:.1f} Hz/s",
                    ),
                    mo.stat(
                        label="Geometric Contact Time",
                        value=f"{pass_geom['max_ergodic_min_per_day']:.1f} min/day",
                        caption=f"Ergodic average at φ = {pass_geom['gs_lat_deg']:.1f}°",
                    ),
                ],
                justify="start",
                wrap=True,
            ),
        ]
    )
    return pass_geom, rx_nf_db


@app.cell
def _(mo):
    # --- Section 2 UI: Optimization Targets & Reliability Constraints ---
    fade_margin_slider = mo.ui.slider(
        start=0.0,
        stop=15.0,
        step=0.5,
        value=6.0,
        label="Required Fade Margin (dB)",
        show_value=True,
    )
    payload_slider = mo.ui.slider(
        start=8,
        stop=255,
        step=1,
        value=32,
        label="Packet Payload Size (Bytes)",
        show_value=True,
    )
    doppler_mode_select = mo.ui.dropdown(
        options={
            "Uncompensated (±BW/2 Passband)": "half_bw",
            "Wide Receiver Tolerance (±BW)": "full_bw",
            "Ground AFC / Doppler Tracked": "tracked",
        },
        value="Uncompensated (±BW/2 Passband)",
        label="Static Doppler Handling",
    )
    objective_select = mo.ui.dropdown(
        options={
            "Balanced (Pass Coverage × Daily Data)": "balanced",
            "Max Reliability (Coverage & Safety Margins)": "reliability",
            "Max Data Link (Daily Downlink kB/day)": "throughput",
        },
        value="Balanced (Pass Coverage × Daily Data)",
        label="Optimization Priority",
    )
    return (
        doppler_mode_select,
        fade_margin_slider,
        objective_select,
        payload_slider,
    )


@app.cell
def _(
    LORA_BWS_KHZ,
    LORA_CRS,
    LORA_SFS,
    doppler_mode_select,
    evaluate_lora_mode,
    fade_margin_slider,
    objective_select,
    pass_geom,
    payload_slider,
    rx_nf_db,
):
    # Exhaustively evaluate all 96 (BW, SF, CR) combinations
    req_fade_margin_db = float(fade_margin_slider.value)
    payload_bytes = int(payload_slider.value)
    _doppler_mode = doppler_mode_select.value
    _objective_mode = objective_select.value

    all_modes = [
        evaluate_lora_mode(
            bw_khz=_bw,
            sf=_sf,
            cr_label=_cr_lbl,
            cr_val=_cr_val,
            payload_bytes=payload_bytes,
            rx_nf_db=rx_nf_db,
            req_fade_margin_db=req_fade_margin_db,
            doppler_mode=_doppler_mode,
            pass_geom=pass_geom,
        )
        for _bw in LORA_BWS_KHZ
        for _sf in LORA_SFS
        for _cr_lbl, _cr_val in LORA_CRS.items()
    ]

    best_reliability_mode = max(
        all_modes,
        key=lambda m: (
            round(m["pass_coverage_pct"], 2),
            min(
                m["horizon_margin_db"] - req_fade_margin_db,
                10.0 * (m["drift_safety_ratio"] - 1.0),
            ),
            m["daily_kb"],
        ),
    )
    best_throughput_mode = max(
        all_modes,
        key=lambda m: (
            m["daily_kb"],
            m["pass_coverage_pct"],
            m["horizon_margin_db"],
        ),
    )
    best_balanced_mode = max(
        all_modes,
        key=lambda m: (
            1 if m["pass_coverage_pct"] >= 99.5 else 0,
            (m["pass_coverage_pct"] / 100.0) ** 2 * m["daily_kb"],
            m["horizon_margin_db"],
        ),
    )

    if _objective_mode == "reliability":
        optimal_mode = best_reliability_mode
    elif _objective_mode == "throughput":
        optimal_mode = best_throughput_mode
    else:
        optimal_mode = best_balanced_mode

    return (
        all_modes,
        best_balanced_mode,
        best_reliability_mode,
        best_throughput_mode,
        optimal_mode,
        payload_bytes,
        req_fade_margin_db,
    )


@app.cell
def _(
    LORA_BWS_KHZ,
    LORA_SFS,
    all_modes,
    best_balanced_mode,
    best_reliability_mode,
    best_throughput_mode,
    doppler_mode_select,
    fade_margin_slider,
    mo,
    np,
    objective_select,
    optimal_mode,
    payload_slider,
    plt,
    req_fade_margin_db,
):
    def _render_solver_plots():
        cov_grid = np.zeros((len(LORA_SFS), len(LORA_BWS_KHZ)))
        cell_annotations = []

        for i, sf in enumerate(LORA_SFS):
            row_annot = []
            for j, bw in enumerate(LORA_BWS_KHZ):
                cell_mode = next(
                    m
                    for m in all_modes
                    if m["sf"] == sf and m["bw_khz"] == bw and m["cr_label"] == "4/5 (CR=1)"
                )
                cov_grid[i, j] = cell_mode["pass_coverage_pct"]
                if cell_mode["pass_coverage_pct"] >= 99.5:
                    tag = f"100% pass\n{cell_mode['daily_kb']:.1f} kB/d"
                elif cell_mode["pass_coverage_pct"] > 0.0:
                    tag = f"{cell_mode['pass_coverage_pct']:.0f}% ({cell_mode['bottleneck']})\n{cell_mode['daily_kb']:.1f} kB/d"
                else:
                    tag = f"0% Pass\n({cell_mode['bottleneck']})"
                row_annot.append(tag)
            cell_annotations.append(row_annot)

        fig, (ax_hm, ax_pareto) = plt.subplots(1, 2, figsize=(12.0, 5.0))

        # Left: 6x4 (SF x BW) Usable Pass Coverage & Daily Data Heatmap
        im = ax_hm.imshow(cov_grid, cmap="RdYlGn", vmin=0.0, vmax=100.0, aspect="auto")
        ax_hm.set_xticks(range(len(LORA_BWS_KHZ)))
        ax_hm.set_xticklabels([f"{bw:g} kHz" for bw in LORA_BWS_KHZ])
        ax_hm.set_yticks(range(len(LORA_SFS)))
        ax_hm.set_yticklabels([f"SF{sf}" for sf in LORA_SFS])
        ax_hm.set_xlabel("LoRa Bandwidth $BW$")
        ax_hm.set_ylabel("Spreading Factor $SF$")
        ax_hm.set_title("Usable Pass Coverage (%) & Daily Yield (CR 4/5)")

        for i in range(len(LORA_SFS)):
            for j in range(len(LORA_BWS_KHZ)):
                is_opt = (
                    LORA_SFS[i] == optimal_mode["sf"]
                    and LORA_BWS_KHZ[j] == optimal_mode["bw_khz"]
                )
                txt_color = "black" if cov_grid[i, j] >= 25.0 else "white"
                ax_hm.text(
                    j,
                    i,
                    cell_annotations[i][j],
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    fontweight="bold" if is_opt else "normal",
                    color=txt_color,
                )
                if is_opt:
                    ax_hm.add_patch(
                        plt.Rectangle(
                            (j - 0.48, i - 0.48),
                            0.96,
                            0.96,
                            fill=False,
                            edgecolor="navy",
                            linewidth=2.8,
                        )
                    )

        plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04, label="Usable Pass Coverage (%)")

        # Right: Reliability (Usable Pass Coverage %) vs. Daily Data Volume Pareto Plot
        viable = [m for m in all_modes if m["pass_coverage_pct"] > 0]
        if viable:
            sc = ax_pareto.scatter(
                [m["pass_coverage_pct"] for m in viable],
                [m["daily_kb"] for m in viable],
                c=[m["sf"] for m in viable],
                cmap="viridis",
                s=55,
                alpha=0.8,
                edgecolors="black",
                linewidth=0.5,
                zorder=3,
            )
            plt.colorbar(sc, ax=ax_pareto, fraction=0.046, pad=0.04, label="Spreading Factor (SF)")

        ax_pareto.scatter(
            [optimal_mode["pass_coverage_pct"]],
            [optimal_mode["daily_kb"]],
            color="crimson",
            s=150,
            marker="*",
            zorder=5,
            label=(
                f"Selected Optimal: SF{optimal_mode['sf']} / {optimal_mode['bw_khz']:g} kHz / "
                f"{optimal_mode['cr_label'].split()[0]}"
            ),
        )
        ax_pareto.set_title("Reliability vs. Data Link Trade-Off (All 96 Modes)")
        ax_pareto.set_xlabel("Usable Pass Coverage (%)")
        ax_pareto.set_ylabel("Daily Downlink Capacity (kB/day)")
        ax_pareto.set_xlim(-5, 105)
        ax_pareto.set_yscale("symlog", linthresh=1.0)
        ax_pareto.grid(True, linestyle="--", alpha=0.5)
        ax_pareto.legend(loc="upper left", fontsize=8.5)

        plt.tight_layout()
        return fig

    _opt_kind = (
        "success"
        if optimal_mode["pass_coverage_pct"] >= 99.5
        else ("warn" if optimal_mode["pass_coverage_pct"] > 0 else "danger")
    )
    _summary_md = (
        f"**Solver Recommendation**: **SF{optimal_mode['sf']}**, "
        f"**BW = {optimal_mode['bw_khz']:g} kHz**, **{optimal_mode['cr_label']}** — closes "
        f"**{optimal_mode['pass_coverage_pct']:.1f}%** of the geometric pass "
        f"($d \\in [{optimal_mode['d_min_km']:.0f}, {optimal_mode['d_max_km']:.0f}]\\text{{ km}}$), "
        f"delivering **{optimal_mode['daily_kb']:.1f} kB/day** "
        f"(**{optimal_mode['daily_packets']:.0f} packets/day** at {optimal_mode['goodput_bps']:.0f} bps net goodput) "
        f"with **{optimal_mode['horizon_margin_db']:+.1f} dB** horizon link margin "
        f"(req: $\\ge {req_fade_margin_db:.1f}\\text{{ dB}}$)."
    )

    mo.vstack(
        [
            mo.md(
                f"""
                ## 2. LoRa Parameter Solver (Reliability & Data Link Optimization)

                Set your link reliability requirements below. The solver exhaustively tests all **96 `(BW, SF, CR)`** modes against **Link Margin**, **Static Doppler**, and **Zenith Doppler Drift Rate**:

                {mo.hstack([fade_margin_slider, payload_slider], justify="start", wrap=True)}
                {mo.hstack([doppler_mode_select, objective_select], justify="start", wrap=True)}
                """
            ),
            mo.callout(mo.md(_summary_md), kind=_opt_kind),
            mo.hstack(
                [
                    mo.stat(
                        label="Best Balanced Mode",
                        value=f"SF{best_balanced_mode['sf']} / {best_balanced_mode['bw_khz']:g} kHz",
                        caption=f"{best_balanced_mode['cr_label'].split()[0]} → {best_balanced_mode['pass_coverage_pct']:.0f}% cov, {best_balanced_mode['daily_kb']:.1f} kB/day",
                    ),
                    mo.stat(
                        label="Max Reliability Mode",
                        value=f"SF{best_reliability_mode['sf']} / {best_reliability_mode['bw_khz']:g} kHz",
                        caption=f"{best_reliability_mode['pass_coverage_pct']:.0f}% cov, {best_reliability_mode['horizon_margin_db']:+.1f} dB horizon margin",
                    ),
                    mo.stat(
                        label="Max Data Link Mode",
                        value=f"SF{best_throughput_mode['sf']} / {best_throughput_mode['bw_khz']:g} kHz",
                        caption=f"{best_throughput_mode['daily_kb']:.1f} kB/day ({best_throughput_mode['pass_coverage_pct']:.0f}% pass cov)",
                    ),
                ],
                justify="start",
                wrap=True,
            ),
            _render_solver_plots(),
        ]
    )
    return


@app.cell
def _(LORA_BWS_KHZ, LORA_CRS, mo):
    # --- Section 3 UI: Active Mode Selector (Auto Solver vs. Manual Override) ---
    use_solver_switch = mo.ui.switch(
        value=True,
        label="Auto-Select Solver's Optimal LoRa Mode",
    )

    bw_manual = mo.ui.dropdown(
        options={f"{bw:g} kHz": bw for bw in LORA_BWS_KHZ},
        value="125 kHz",
        label="Manual Bandwidth $BW$",
    )
    sf_manual = mo.ui.slider(
        start=7,
        stop=12,
        step=1,
        value=7,
        label="Manual Spreading Factor $SF$",
        show_value=True,
    )
    cr_manual = mo.ui.dropdown(
        options=LORA_CRS,
        value="4/5 (CR=1)",
        label="Manual Coding Rate $CR$",
    )
    return bw_manual, cr_manual, sf_manual, use_solver_switch


@app.cell
def _(
    LORA_CRS,
    all_modes,
    bw_manual,
    cr_manual,
    mo,
    optimal_mode,
    sf_manual,
    use_solver_switch,
):
    # Resolve active LoRa mode (either solver optimal or manual override)
    if use_solver_switch.value:
        active_mode = optimal_mode
        _mode_source = "Solver Optimal (Auto)"
    else:
        _cr_lbl = next(k for k, v in LORA_CRS.items() if abs(v - cr_manual.value) < 1e-9)
        active_mode = next(
            m
            for m in all_modes
            if m["bw_khz"] == bw_manual.value
            and m["sf"] == sf_manual.value
            and m["cr_label"] == _cr_lbl
        )
        _mode_source = "Manual Override"

    _manual_controls = (
        mo.md("*Toggle off **Auto-Select** above to manually override `BW`, `SF`, and `CR`.*")
        if use_solver_switch.value
        else mo.hstack([bw_manual, sf_manual, cr_manual], justify="start", wrap=True)
    )

    mo.vstack(
        [
            mo.md("## 3. Active LoRa Configuration (Driving Pass Diagnostics)"),
            use_solver_switch,
            _manual_controls,
            mo.hstack(
                [
                    mo.stat(
                        label=f"Active Mode ({_mode_source})",
                        value=f"SF{active_mode['sf']} / {active_mode['bw_khz']:g} kHz",
                        caption=f"Coding Rate: {active_mode['cr_label']}",
                    ),
                    mo.stat(
                        label="Net Payload Goodput",
                        value=f"{active_mode['goodput_bps']:.0f} bps"
                        if active_mode["goodput_bps"] < 1000
                        else f"{active_mode['goodput_bps'] / 1000.0:.2f} kbps",
                        caption=f"Raw: {active_mode['nominal_bps']:.0f} bps (ToA = {active_mode['t_packet_s'] * 1000:.0f} ms)",
                    ),
                    mo.stat(
                        label="Usable Pass Window",
                        value=f"{active_mode['pass_coverage_pct']:.1f}%",
                        caption=f"d ∈ [{active_mode['d_min_km']:.0f}, {active_mode['d_max_km']:.0f}] km ({active_mode['usable_min_per_day']:.1f} min/day)",
                    ),
                    mo.stat(
                        label="Daily Downlink Volume",
                        value=f"{active_mode['daily_kb']:.1f} kB/day",
                        caption=f"~{active_mode['daily_packets']:.0f} packets/day",
                    ),
                ],
                justify="start",
                wrap=True,
            ),
        ]
    )
    return (active_mode,)


@app.cell
def _(
    AutoMinorLocator,
    LORA_SNR_THRESHOLDS_DB,
    active_mode,
    mo,
    pass_geom,
    plt,
    req_fade_margin_db,
    rx_nf_db,
):
    # --- Section 4A: Link Margin & Ergodic Contact Diagnostics ---
    def _render_link_and_contact():
        range_km = pass_geom["range_km"]
        pr_dbm = pass_geom["pr_dbm"]
        alt_km = pass_geom["alt_km"]
        horiz_km = pass_geom["horizon_km"]

        sf_val = active_mode["sf"]
        bw_khz = active_mode["bw_khz"]
        thermal_noise_dbm = active_mode["thermal_noise_dbm"]
        lora_sens_dbm = active_mode["lora_sens_dbm"]
        req_rx_dbm = lora_sens_dbm + req_fade_margin_db

        fig, (ax_pr, ax_erg) = plt.subplots(1, 2, figsize=(12.0, 4.8))

        # Left subplot: Received Power vs. Slant Range & LoRa Sensitivity Thresholds
        ax_pr.plot(range_km, pr_dbm, color="#1f77b4", lw=2.4, zorder=4, label="Received Power ($P_r$)")
        ax_pr.scatter([alt_km, horiz_km], [pr_dbm[0], pr_dbm[-1]], color="#1f77b4", s=45, zorder=5)

        if active_mode["pass_coverage_pct"] > 0:
            ax_pr.axvspan(
                active_mode["d_min_km"],
                active_mode["d_max_km"],
                color="limegreen",
                alpha=0.14,
                label=f"Usable Pass Window ({active_mode['d_min_km']:.0f}–{active_mode['d_max_km']:.0f} km)",
            )

        for sf, snr_lim in sorted(LORA_SNR_THRESHOLDS_DB.items()):
            sens = thermal_noise_dbm + snr_lim
            if sf == sf_val:
                ax_pr.axhline(
                    sens,
                    color="crimson",
                    linestyle="-",
                    lw=1.8,
                    label=f"Active SF{sf} Floor ({sens:.1f} dBm)",
                )
                ax_pr.axhline(
                    req_rx_dbm,
                    color="darkorange",
                    linestyle="-.",
                    lw=1.6,
                    label=f"SF{sf} + {req_fade_margin_db:.1f} dB Fade Margin ({req_rx_dbm:.1f} dBm)",
                )
            else:
                ax_pr.axhline(sens, color="gray", linestyle="--", lw=0.9, alpha=0.5)

        ax_pr.set_title(
            f"Received Power & Sensitivity vs. Slant Range\n"
            f"(SF{sf_val}, BW = {bw_khz:g} kHz, NF = {rx_nf_db:.1f} dB)"
        )
        ax_pr.set_xlabel("Slant Range $d$ (km)")
        ax_pr.set_ylabel("Received Power (dBm)")
        ax_pr.grid(True, linestyle="--", alpha=0.5)
        ax_pr.legend(loc="upper right", fontsize=7.8)

        ax_snr = ax_pr.secondary_yaxis(
            "right",
            functions=(
                lambda p_dbm: p_dbm - thermal_noise_dbm,
                lambda snr_db: snr_db + thermal_noise_dbm,
            ),
        )
        ax_snr.set_ylabel("Received SNR (dB)")

        # Right subplot: Ergodic Contact Time vs. Slant Range with Active Window Shaded
        ax_erg.plot(
            pass_geom["erg_ranges_km"],
            pass_geom["erg_min_per_day"],
            color="#2ca02c",
            lw=2.2,
            label="Cumulative Ergodic Contact (min/day)",
        )
        ax_erg.axvline(
            horiz_km,
            color="tab:red",
            linestyle="--",
            lw=1.4,
            label=f"Radar Horizon ({horiz_km:.0f} km)",
        )
        if active_mode["pass_coverage_pct"] > 0:
            ax_erg.axvspan(
                active_mode["d_min_km"],
                active_mode["d_max_km"],
                color="limegreen",
                alpha=0.18,
                label=f"Active Usable Range ({active_mode['usable_min_per_day']:.1f} min/day)",
            )
        ax_erg.set_title(
            f"Ergodic Contact Time vs. Slant Range\n"
            f"(h = {alt_km:.0f} km, i = {pass_geom['inc_deg']:.1f}°, φ = {pass_geom['gs_lat_deg']:.1f}°)"
        )
        ax_erg.set_xlabel("Maximum Slant Range $d$ (km)")
        ax_erg.set_ylabel("Average Contact Time (minutes/day)")
        ax_erg.xaxis.set_minor_locator(AutoMinorLocator())
        ax_erg.yaxis.set_minor_locator(AutoMinorLocator())
        ax_erg.grid(True, linestyle="--", alpha=0.5)
        ax_erg.set_xlim(0, pass_geom["erg_ranges_km"][-1])
        ax_erg.set_ylim(bottom=0)
        ax_erg.legend(loc="upper left", fontsize=8)

        plt.tight_layout()
        return fig

    mo.vstack(
        [
            mo.md("## 4. Detailed Pass Diagnostics (Active Mode)"),
            _render_link_and_contact(),
        ]
    )
    return


@app.cell
def _(AutoMinorLocator, active_mode, mo, pass_geom, payload_bytes, plt):
    # --- Section 4B: Doppler Shift & Doppler Drift Rate Diagnostics ---
    def _render_doppler_diagnostics():
        range_km = pass_geom["range_km"]
        alt_km = pass_geom["alt_km"]
        horiz_km = pass_geom["horizon_km"]

        doppler_khz = pass_geom["doppler_hz"] / 1e3
        doppler_max_khz = pass_geom["doppler_shift_max_hz"] / 1e3
        bw_khz = active_mode["bw_khz"]

        doppler_rate_mag_hz_s = pass_geom["doppler_rate_mag_hz_s"]
        max_rate_hz_s = pass_geom["max_doppler_rate_hz_s"]

        pkt_limit = active_mode["pkt_drift_limit_hz_s"]
        sym_limit = active_mode["sym_chirp_limit_hz_s"]

        fig, (ax_shift, ax_rate) = plt.subplots(1, 2, figsize=(12.0, 4.8))

        # Left: Static Doppler Shift vs. Slant Range
        ax_shift.plot(range_km, doppler_khz, color="tab:blue", lw=2.2, label="Approaching (+Δf)")
        ax_shift.plot(
            range_km,
            -doppler_khz,
            color="tab:orange",
            lw=2.2,
            linestyle="--",
            label="Receding (-Δf)",
        )
        ax_shift.axhline(
            0.5 * bw_khz,
            color="crimson",
            linestyle=":",
            lw=1.5,
            label=f"±BW/2 Limit (±{0.5 * bw_khz:.1f} kHz)",
        )
        ax_shift.axhline(-0.5 * bw_khz, color="crimson", linestyle=":", lw=1.5)
        ax_shift.scatter([alt_km], [0.0], color="tab:green", s=50, zorder=5)
        ax_shift.scatter(
            [horiz_km, horiz_km],
            [doppler_max_khz, -doppler_max_khz],
            color="tab:red",
            s=50,
            zorder=5,
            label=f"Horizon (±{doppler_max_khz:.2f} kHz)",
        )
        ax_shift.set_title(f"Doppler Shift vs. Slant Range (BW = {bw_khz:g} kHz)")
        ax_shift.set_xlabel("Slant Range $d$ (km)")
        ax_shift.set_ylabel("Doppler Shift $\\Delta f$ (kHz)")
        ax_shift.xaxis.set_minor_locator(AutoMinorLocator())
        ax_shift.yaxis.set_minor_locator(AutoMinorLocator())
        ax_shift.grid(True, linestyle="--", alpha=0.5)
        ax_shift.legend(loc="best", fontsize=8)

        # Right: Log-Scale Doppler Rate Magnitude vs. Active LoRa Limits
        ax_rate.plot(
            range_km[:-1],
            doppler_rate_mag_hz_s[:-1],
            color="tab:cyan",
            lw=2.2,
            label=r"Pass $\left|\frac{d(\Delta f)}{dt}\right|$ (Hz/s)",
        )
        ax_rate.axhline(
            pkt_limit,
            color="darkorange",
            linestyle="-.",
            lw=1.8,
            label=f"Intra-Packet Drift Limit ({pkt_limit:.1f} Hz/s, {payload_bytes}B)",
        )
        ax_rate.axhline(
            sym_limit,
            color="crimson",
            linestyle=":",
            lw=1.8,
            label=f"Intra-Symbol Chirp Limit ({sym_limit:.1f} Hz/s)",
        )
        ax_rate.scatter(
            [alt_km],
            [max_rate_hz_s],
            color="tab:green",
            s=55,
            zorder=5,
            label=f"Zenith Peak ({max_rate_hz_s:.1f} Hz/s)",
        )
        if active_mode["zenith_blind_spot_km"] > 0:
            ax_rate.axvspan(
                alt_km,
                active_mode["d_min_km"],
                color="crimson",
                alpha=0.18,
                label=f"Zenith Blind Spot (d < {active_mode['d_min_km']:.0f} km)",
            )

        ax_rate.set_yscale("log")
        ax_rate.set_ylim(
            bottom=max(0.5, min(max_rate_hz_s, pkt_limit) * 0.05),
            top=max(sym_limit, max_rate_hz_s) * 2.5,
        )
        ax_rate.set_title(
            f"Doppler Drift Rate vs. LoRa Limits\n"
            f"(SF{active_mode['sf']}, BW = {bw_khz:g} kHz, {'LDRO On' if active_mode['ldro_enabled'] else 'LDRO Off'})"
        )
        ax_rate.set_xlabel("Slant Range $d$ (km)")
        ax_rate.set_ylabel(r"Drift Rate $\left|\frac{d(\Delta f)}{dt}\right|$ (Hz/s)")
        ax_rate.xaxis.set_minor_locator(AutoMinorLocator())
        ax_rate.grid(True, which="both", linestyle="--", alpha=0.45)
        ax_rate.legend(loc="best", fontsize=8)

        plt.tight_layout()
        return fig

    _max_pkt_drift_hz = pass_geom["max_doppler_rate_hz_s"] * active_mode["t_packet_s"]
    _max_pkt_bins = _max_pkt_drift_hz / active_mode["lora_bin_hz"]

    mo.vstack(
        [
            _render_doppler_diagnostics(),
            mo.hstack(
                [
                    mo.stat(
                        label="Horizon Margin",
                        value=f"{active_mode['horizon_margin_db']:+.2f} dB",
                        caption=f"Zenith Margin: {active_mode['zenith_margin_db']:+.2f} dB",
                    ),
                    mo.stat(
                        label="Doppler / BW Ratio",
                        value=f"{(pass_geom['doppler_shift_max_hz'] / active_mode['bw_hz']) * 100:.1f}%",
                        caption=f"±{pass_geom['doppler_shift_max_hz'] / 1e3:.2f} kHz vs. {active_mode['bw_khz']:g} kHz BW",
                    ),
                    mo.stat(
                        label=f"Zenith Drift per {payload_bytes}B Packet",
                        value=f"{_max_pkt_drift_hz:.1f} Hz ({_max_pkt_bins:.2f} bins)",
                        caption=f"Limit: {active_mode['drift_margin_hz']:.1f} Hz (bin = {active_mode['lora_bin_hz']:.1f} Hz)",
                    ),
                    mo.stat(
                        label="Mode Limiting Factor",
                        value=active_mode["bottleneck"],
                        caption=f"Drift safety factor: {active_mode['drift_safety_ratio']:.2f}×",
                    ),
                ],
                justify="start",
                wrap=True,
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
