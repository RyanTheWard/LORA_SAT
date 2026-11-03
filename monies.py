import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    return AutoMinorLocator, mo, np, plt


@app.cell
def _(mo):
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

    orbits_slider = mo.ui.slider(
        start=1,
        stop=15,
        step=1,
        value=3,
        label="Number of Orbits",
        show_value=True,
    )

    proj_select = mo.ui.dropdown(
        options={
            "Equirectangular": "Equirectangular",
            "Natural Earth": "natural earth",
            "Orthographic (3D Globe)": "orthographic",
            "Mercator": "mercator",
            "Mollweide": "mollweide",
        },
        value="Equirectangular",
        label="Map Projection",
    )
    return alt_slider, inc_slider, orbits_slider, proj_select


@app.cell
def _(mo):
    # UI controls configured for a 433 MHz CubeSat downlink
    pt_input = mo.ui.number(
        value=0.1,
        start=0.01,
        stop=5,
        step=0.005,
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
        stop=8.0,
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

    d_slider = mo.ui.slider(
        start=400,
        stop=3000,
        value=800,
        step=25,
        label="Slant Range $d$ (km)",
        show_value=True,
    )
    return d_slider, freq_input, gr_input, gt_input, pt_input


@app.cell(hide_code=True)
def _(alt_slider, inc_slider, mo, orbits_slider, proj_select):
    mo.md(f"""
    ### Orbit Controls

    Adjust the satellite altitude, inclination, and number of passes to explore the ground track geometry:

    {mo.hstack([alt_slider, inc_slider], justify="start")}
    {mo.hstack([orbits_slider, proj_select], justify="start")}
    """)
    return


@app.cell(hide_code=True)
def _(d_slider, freq_input, gr_input, gt_input, mo, pt_input):
    mo.md(f"""
    ### Link Configuration

    Adjust the parameters below to evaluate the link budget for the CubeSat pass:

    {mo.hstack([pt_input, freq_input], justify="start")}
    {mo.hstack([gt_input, gr_input], justify="start")}
    {d_slider}
    """)
    return


@app.cell
def _(d_slider, freq_input, gr_input, gt_input, mo, np, pt_input):
    # Conversions and Friis formula calculations
    c_light = 299_792_458.0  # speed of light in m/s
    freq_hz = freq_input.value * 1e6
    wavelength = c_light / freq_hz

    # Linear gains from dBi
    gt_linear = 10.0 ** (gt_input.value / 10.0)
    gr_linear = 10.0 ** (gr_input.value / 10.0)

    # Distance in meters
    dist_km = d_slider.value
    dist_m = dist_km * 1000.0
    power_tx_w = pt_input.value

    # Friis Transmission Formula
    pr_watts = power_tx_w * gt_linear * gr_linear * ((wavelength / (4.0 * np.pi * dist_m)) ** 2)
    pr_dbm = 10.0 * np.log10(pr_watts * 1000.0)
    fspl_db = 20.0 * np.log10((4.0 * np.pi * dist_m) / wavelength)

    def pr_dbm_rec(d):
        return  10.0 * np.log10(power_tx_w * gt_linear * gr_linear * ((wavelength / (4.0 * np.pi * d)) ** 2) * 1000.0)


    mo.hstack(
        [
            mo.stat(
                label="Received Power",
                value=f"{pr_dbm:.2f} dBm",
                caption=f"{pr_watts * 1e12:.2f} pW",
            ),
            mo.stat(
                label="Free-Space Path Loss",
                value=f"{fspl_db:.2f} dB",
                caption=f"λ = {wavelength * 100:.1f} cm",
            ),
            mo.stat(
                label="EIRP",
                value=f"{(10.0 * np.log10(power_tx_w * 1000.0) + gt_input.value):.2f} dBm",
                caption=f"{power_tx_w * gt_linear:.2f} W equivalent",
            ),
            mo.stat(
                label="Current Slant Range",
                value=f"{dist_km} km",
            ),
        ],
        justify="start",
    )
    return (
        c_light,
        dist_km,
        freq_hz,
        fspl_db,
        gr_linear,
        gt_linear,
        power_tx_w,
        pr_dbm,
        wavelength,
    )


@app.cell
def _(
    dist_km,
    fspl_db,
    gr_linear,
    gt_linear,
    np,
    plt,
    power_tx_w,
    pr_dbm,
    wavelength,
):
    # Plot link parameters across the full slant range of a LEO pass
    range_km = np.linspace(400, 2500, 500)
    range_m = range_km * 1000.0

    pr_range_w = power_tx_w * gt_linear * gr_linear * ((wavelength / (4.0 * np.pi * range_m)) ** 2)
    pr_range_dbm = 10.0 * np.log10(pr_range_w * 1000.0)
    fspl_range_db = 20.0 * np.log10((4.0 * np.pi * range_m) / wavelength)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Plot 1: Received Power (dBm) vs Slant Range
    ax1.plot(range_km, pr_range_dbm, color="#1f77b4", lw=2, label="Received Power ($P_r$)")
    ax1.scatter([dist_km], [pr_dbm], color="red", s=60, zorder=5, label=f"Selected: {dist_km} km ({pr_dbm:.1f} dBm)")
    ax1.axhline(-115, color="gray", linestyle=":", label="Typical GFSK Sens. (-115 dBm)")
    ax1.axhline(-125, color="purple", linestyle="--", alpha=0.7, label="LoRa Sens. (-125 dBm)")
    ax1.set_title("Received Power vs. Slant Range (433 MHz)")
    ax1.set_xlabel("Slant Range $d$ (km)")
    ax1.set_ylabel("Received Power (dBm)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8)

    # Plot 2: Free-Space Path Loss (dB) vs Slant Range
    ax2.plot(range_km, fspl_range_db, color="#e6550d", lw=2, label="Free-Space Path Loss")
    ax2.scatter([dist_km], [fspl_db], color="red", s=60, zorder=5, label=f"Loss: {fspl_db:.1f} dB")
    ax2.set_title("Free-Space Path Loss (FSPL) vs. Slant Range")
    ax2.set_xlabel("Slant Range $d$ (km)")
    ax2.set_ylabel("Path Loss (dB)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    # LoRa Modulation UI Controls
    bw_select = mo.ui.dropdown(
        options={
            "62.5 kHz": 62.5,
            "125 kHz": 125.0,
            "250 kHz": 250.0,
            "500 kHz": 500.0,
        },
        value="125 kHz",
        label="Bandwidth $BW$",
    )

    sf_slider = mo.ui.slider(
        start=7,
        stop=12,
        step=1,
        value=7,
        label="Spreading Factor $SF$",
        show_value=True,
    )

    cr_select = mo.ui.dropdown(
        options={
            "4/5 (CR=1)": 4 / 5,
            "4/6 (CR=2)": 4 / 6,
            "4/7 (CR=3)": 4 / 7,
            "4/8 (CR=4)": 4 / 8,
        },
        value="4/5 (CR=1)",
        label="Coding Rate $CR$",
    )
    return bw_select, cr_select, sf_slider


@app.cell
def _(bw_select, cr_select, mo, sf_slider):
    # LoRa Modulation Calculations & Display
    bw_khz = bw_select.value
    bw_hz = bw_khz * 1e3
    sf_val = sf_slider.value
    cr_val = cr_select.value

    # Symbol duration: Ts = (2^SF) / BW
    t_symbol_ms = ((2**sf_val) / bw_hz) * 1000.0

    # Nominal LoRa Bit Rate: Rb = SF * (BW / 2^SF) * CR
    bit_rate_bps = sf_val * (bw_hz / (2**sf_val)) * cr_val

    # LoRa receiver sensitivity SNR threshold (empirical standard limits)
    lora_snr_thresholds = {
        7: -7.5,
        8: -10.0,
        9: -12.5,
        10: -15.0,
        11: -17.5,
        12: -20.0,
    }
    snr_limit_db = lora_snr_thresholds.get(sf_val, -7.5)

    mo.vstack(
        [
            mo.md(
                f"""
    ### LoRa Modulation Parameters

    Select the physical layer parameters for the CubeSat LoRa downlink:

    {mo.hstack([bw_select, sf_slider, cr_select], justify="start", gap=2)}
    """
            ),
            mo.hstack(
                [
                    mo.stat(
                        label="Nominal Bit Rate",
                        value=f"{bit_rate_bps:.2f} bps"
                        if bit_rate_bps < 1000
                        else f"{bit_rate_bps / 1000.0:.2f} kbps",
                        caption=r"$R_b = SF \cdot \frac{BW}{2^{SF}} \cdot CR$",
                    ),
                    mo.stat(
                        label="Symbol Duration",
                        value=f"{t_symbol_ms:.2f} ms",
                        caption=r"$T_s = 2^{SF} / BW$",
                    ),
                    mo.stat(
                        label="LoRa Demod SNR Limit",
                        value=f"{snr_limit_db:.1f} dB",
                        caption=f"Sensitivity floor at SF{sf_val}",
                    ),
                    mo.stat(
                        label="Chirp Rate",
                        value=f"{bw_khz:.1f} kchirps/s",
                        caption="Chip rate = BW",
                    ),
                ],
                justify="start",
            ),
        ]
    )
    return bw_hz, bw_khz


@app.cell
def _(alt_slider, c_light, freq_hz, inc_slider, mo, np, orbits_slider):
    orbit_alt_km = alt_slider.value
    orbit_inc_deg = inc_slider.value
    num_orbits = orbits_slider.value


    MU_EARTH = 3.986004418e14   # m^3/s^2
    R_EARTH  = 6371000.0        # m


    h = orbit_alt_km * 1000.0   # altitude, m
    r_sat = R_EARTH + h         # orbital radius, m

    v_s = np.sqrt(MU_EARTH / r_sat)

    # Maximum line-of-sight speed magnitude, at horizon
    v_los_max = v_s * (R_EARTH / r_sat)

    # Maximum Doppler shift magnitude
    doppler_shift_max = freq_hz * (v_los_max / c_light)

    # Slant distance to horizon
    horizon_distance_km = np.sqrt(r_sat**2 - R_EARTH**2) / 1000.0

    # Ground/arc distance to horizon
    ground_horizon_distance_km = (
        R_EARTH * np.arccos(R_EARTH / r_sat) / 1000.0
    )

    def link_dist(theta):
        return np.sqrt(
            R_EARTH**2
            + r_sat**2
            - 2.0 * R_EARTH * r_sat * np.cos(theta)
        )

    def doppler_theta(theta):
        return -(
            freq_hz
            * v_s
            * R_EARTH
            * np.sin(theta)
            / (c_light * link_dist(theta))
        )

    theta_horizon = np.arccos(R_EARTH / r_sat)

    mo.vstack(
        [
            mo.md("### Orbit & Horizon Geometry"),
            mo.hstack(
                [
                    mo.stat(
                        label="Orbital Velocity ($v_s$)",
                        value=f"{v_s:.0f} m/s",
                        caption=f"{v_s * 3.6:.0f} km/h",
                    ),

                    mo.stat(
                        label="Slant Horizon",
                        value=f"{horizon_distance_km:.0f} km",
                        caption="Max LOS range",
                    ),
                    mo.stat(
                        label="Ground Horizon",
                        value=f"{ground_horizon_distance_km:.0f} km",
                        caption="Footprint arc radius",
                    ),
                    mo.stat(
                        label="Horizon Angle ($\theta_h$)",
                        value=f"{np.degrees(theta_horizon):.2f}°",
                        caption=f"{theta_horizon:.3f} rad",
                    ),
                ],
                justify="start",
                wrap=True,
            ),
        ]
    )
    return doppler_shift_max, horizon_distance_km, orbit_alt_km, orbit_inc_deg


@app.cell(hide_code=True)
def _(bw_hz, bw_khz, doppler_shift_max, mo):

    doppler_khz = doppler_shift_max / 1e3
    bw_ratio_pct = (doppler_shift_max / bw_hz) * 100.0
    within_half_bw = doppler_shift_max <= (bw_hz / 2.0)
    within_full_bw = doppler_shift_max <= bw_hz

    if within_half_bw:
        status_type = "success"
        check_text = (
            f"**Within Channel Limits**: Maximum Doppler shift of **±{doppler_khz:.2f} kHz** "
            f"is within the receiver passband ($$\\pm BW/2 = \\pm{bw_khz / 2:.1f}$$ kHz), "
            f"representing **{bw_ratio_pct:.1f}%** of the total {bw_khz:.1f} kHz bandwidth."
        )
    elif within_full_bw:
        status_type = "warn"
        check_text = (
            f"**Marginal**: Maximum Doppler shift of **±{doppler_khz:.2f} kHz** "
            f"is within the total channel bandwidth ({bw_khz:.1f} kHz), but exceeds $$\\pm BW/2$$. "
            f"Automatic Frequency Correction (AFC) or Doppler tracking is recommended."
        )
    else:
        status_type = "danger"
        check_text = (
            f"**Out of Band**: Maximum Doppler shift of **±{doppler_khz:.2f} kHz** "
            f"exceeds the chosen bandwidth of **{bw_khz:.1f} kHz** ({bw_ratio_pct:.1f}% of BW). "
            f"The signal will shift out of the demodulator passband without active compensation."
        )

    mo.vstack(
        [
            mo.md("### Doppler Shift Verification"),
            mo.hstack(
                [
                    mo.stat(
                        label="Max Doppler Shift",
                        value=f"±{doppler_khz:.2f} kHz",
                        caption=f"{doppler_shift_max:.1f} Hz",
                    ),
                    mo.stat(
                        label="LoRa Bandwidth",
                        value=f"{bw_khz:.1f} kHz",
                        caption=f"{bw_hz:.0f} Hz",
                    ),
                    mo.stat(
                        label="Doppler / BW Ratio",
                        value=f"{bw_ratio_pct:.1f}%",
                        caption=r"$\Delta f_{max} / BW$",
                    ),
                ],
                justify="start",
            ),
            mo.callout(mo.md(check_text), kind=status_type),
        ]
    )
    return


@app.cell
def _(AutoMinorLocator, np, plt):


    def plot_time_in_range(
        altitude_km,
        inclination_deg,
        station_lat_deg,
        station_lon_deg,
        ranges_km=None,
        range_max_km=None,
        n_range_points=80,
        duration_hours=24,
        dt_s=1.0,
        raan_deg=0.0,
        u0_deg=0.0,
        gmst0_deg=0.0,
        min_elevation_deg=0.0,
    ):
        """
        Plot satellite-ground station contact statistics versus
        maximum communication range.

        Returns
        -------
        fig, axes, ranges_km, contact_hours,
        avg_contact_min_per_day, avg_contacts_per_day
        """

        # --------------------------------------------------
        # Constants
        # --------------------------------------------------

        mu = 398600.4418
        Re = 6378.137
        omega_earth = 7.2921159e-5

        # --------------------------------------------------
        # Input validation
        # --------------------------------------------------

        if altitude_km <= 0:
            raise ValueError("Altitude must be positive.")

        if duration_hours <= 0 or dt_s <= 0:
            raise ValueError("Duration and time step must be positive.")

        if not 0 <= inclination_deg <= 180:
            raise ValueError("Inclination must be between 0 and 180 degrees.")

        # --------------------------------------------------
        # Communication range sweep
        # --------------------------------------------------

        if ranges_km is None:
            if range_max_km is None:
                range_max_km = max(5000, altitude_km + 3000)

            ranges_km = np.linspace(
                0, range_max_km, n_range_points
            )
        else:
            ranges_km = np.asarray(ranges_km, dtype=float)

        if ranges_km.ndim != 1 or len(ranges_km) == 0:
            raise ValueError("ranges_km must be a non-empty 1D array.")

        if np.any(ranges_km < 0):
            raise ValueError("Communication ranges cannot be negative.")

        # --------------------------------------------------
        # Simulation time
        # --------------------------------------------------

        duration_s = duration_hours * 3600

        t = np.arange(0, duration_s, dt_s)
        t = np.append(t, duration_s)

        duration_days = duration_s / 86400

        # --------------------------------------------------
        # Circular orbit propagation in ECI
        # --------------------------------------------------

        r_orbit = Re + altitude_km
        n = np.sqrt(mu / r_orbit**3)

        inc = np.radians(inclination_deg)
        raan = np.radians(raan_deg)
        u0 = np.radians(u0_deg)

        u = u0 + n * t

        x_orbit = r_orbit * np.cos(u)
        y_orbit = r_orbit * np.sin(u)

        x_sat = (
            np.cos(raan) * x_orbit
            - np.sin(raan) * np.cos(inc) * y_orbit
        )

        y_sat = (
            np.sin(raan) * x_orbit
            + np.cos(raan) * np.cos(inc) * y_orbit
        )

        z_sat = np.sin(inc) * y_orbit

        # --------------------------------------------------
        # Ground station position in ECEF
        # --------------------------------------------------

        lat = np.radians(station_lat_deg)
        lon = np.radians(station_lon_deg)

        r_gs = Re * np.array([
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat)
        ])

        up = r_gs / Re

        # --------------------------------------------------
        # Transform satellite position from ECI to ECEF
        # --------------------------------------------------

        theta = np.radians(gmst0_deg) + omega_earth * t

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        x_ecef = cos_theta * x_sat + sin_theta * y_sat
        y_ecef = -sin_theta * x_sat + cos_theta * y_sat
        z_ecef = z_sat

        # --------------------------------------------------
        # Satellite-ground station relative position
        # --------------------------------------------------

        rho_x = x_ecef - r_gs[0]
        rho_y = y_ecef - r_gs[1]
        rho_z = z_ecef - r_gs[2]

        distance = np.sqrt(
            rho_x**2 + rho_y**2 + rho_z**2
        )

        # --------------------------------------------------
        # Ground station elevation angle
        # --------------------------------------------------

        vertical_distance = (
            rho_x * up[0]
            + rho_y * up[1]
            + rho_z * up[2]
        )

        elevation = np.degrees(
            np.arcsin(
                np.clip(vertical_distance / distance, -1, 1)
            )
        )

        visible = elevation >= min_elevation_deg

        # --------------------------------------------------
        # Contact statistics for each communication range
        # --------------------------------------------------

        contact_hours = np.zeros(len(ranges_km))
        avg_contact_min_per_day = np.zeros(len(ranges_km))
        avg_contacts_per_day = np.zeros(len(ranges_km))

        for j, X in enumerate(ranges_km):

            in_contact = (distance <= X) & visible

            # ----------------------------------------------
            # Total contact time
            # ----------------------------------------------

            contact_fraction = 0.5 * (
                in_contact[:-1].astype(float)
                + in_contact[1:].astype(float)
            )

            contact_time_s = np.sum(
                contact_fraction * np.diff(t)
            )

            contact_hours[j] = contact_time_s / 3600

            # ----------------------------------------------
            # Average contact minutes per day
            # ----------------------------------------------

            avg_contact_min_per_day[j] = (
                contact_time_s / 60 / duration_days
            )

            # ----------------------------------------------
            # Number of distinct contact events
            # ----------------------------------------------

            # Count transitions from outside to inside.
            # A contact already in progress at t=0
            # is counted as one event.

            contact_starts = (
                in_contact[1:] & ~in_contact[:-1]
            )

            n_contacts = (
                int(in_contact[0])
                + np.count_nonzero(contact_starts)
            )

            avg_contacts_per_day[j] = (
                n_contacts / duration_days
            )

        # --------------------------------------------------
        # Radar horizon
        # --------------------------------------------------

        radar_horizon_km = np.sqrt(
            (Re + altitude_km)**2 - Re**2
        )

        # --------------------------------------------------
        # Plotting
        # --------------------------------------------------

        fig, axes = plt.subplots(
            2, 1,
            figsize=(6, 6)
        )

        plot_data = [
            (avg_contact_min_per_day, "Average Contact Time \n(minutes/day)"),
            (avg_contacts_per_day, "Average Number of Contacts \n(contacts/day)")
        ]

        for ax, (data, ylabel) in zip(axes, plot_data):

            ax.plot(
                ranges_km,
                data,
                linewidth=2
            )

            ax.axvline(
                radar_horizon_km,
                color="tab:red",
                linestyle="--",
                linewidth=1.5,
                label=f"Radar horizon ({radar_horizon_km:.0f} km)"
            )

            ax.set_ylabel(ylabel)

            # Minor ticks
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())

            # Major and minor gridlines
            ax.grid(which="major", alpha=0.3)
            ax.grid(
                which="minor",
                color="lightgrey",
                linestyle=":",
                linewidth=0.6,
                alpha=0.7
            )

            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

            ax.legend(loc="best")

        axes[0].set_title(
            f"Satellite-Ground Station Contact Statistics\n"
            f"Altitude = {altitude_km} km, "
            f"Inclination = {inclination_deg}°"
        )

        axes[-1].set_xlabel("Maximum Communication Range (km)")

        plt.tight_layout()
        plt.show()

        return (
            fig,
            axes,
            ranges_km,
            contact_hours,
            avg_contact_min_per_day,
            avg_contacts_per_day
        )


    return (plot_time_in_range,)


@app.cell
def _(horizon_distance_km, orbit_alt_km, orbit_inc_deg, plot_time_in_range):
    fig_ct, axes_ct, ranges_ct, hours_ct, min_per_day_ct, contacts_per_day_ct = plot_time_in_range(
        altitude_km=orbit_alt_km,
        inclination_deg=orbit_inc_deg,
        station_lat_deg=51.0447,
        station_lon_deg=-114.0719,
        range_max_km=horizon_distance_km,
        duration_hours=24*30,
        dt_s=1.5
    )


    fig_ct
    return


if __name__ == "__main__":
    app.run()
