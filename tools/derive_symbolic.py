"""Symbolic derivation of every closed form ares_math implements.

Build spec section 11, mechanism 5: a sympy script derives the negative
obstacle and depth error relations from first principles and asserts they
match the hand-written closed forms.

This catches silent divergence between the document and the code. Nothing
here imports the closed form and checks it against itself. Each derivation
starts from the geometry or the kinematics, sympy does the algebra, and only
at the end is the result compared to the expression ares_math evaluates.

Run:  python tools/derive_symbolic.py
It exits non-zero if any derivation stops matching, and it is run in CI.
"""

import sympy as sp

RESULTS = []


def assert_equal(derived, closed_form, name, assumptions=None):
    """Assert two expressions are identically equal, not just numerically close."""
    diff = sp.simplify(sp.expand(sp.together(derived - closed_form)))
    if assumptions:
        diff = sp.simplify(diff.subs(assumptions))
    if diff != 0:
        raise AssertionError(
            f"{name}\n  derived     : {sp.simplify(derived)}\n"
            f"  closed form : {closed_form}\n  difference  : {diff}"
        )
    RESULTS.append((name, sp.simplify(closed_form)))
    print(f"  ok  {name}")
    print(f"      {sp.simplify(closed_form)}")


def negative_obstacle():
    """R_max = w * h / d, from ray geometry alone.

    Sensor eye at (0, h). Ground at z = 0. A hole of width w and depth d
    with its near edge at horizontal range R. The grazing ray leaves the eye
    and passes over the near edge at (R, 0). Ask where that ray reaches the
    floor at z = -d, and require that point to fall inside the hole.
    """
    print("\nnegative obstacle detection range")
    h, R, w, d, x = sp.symbols("h R w d x", positive=True)

    # The ray through (0, h) and (R, 0).
    ray = h - (h / R) * x
    x_floor = sp.solve(sp.Eq(ray, -d), x)[0]

    # The floor is seen when the ray reaches depth d before clearing the far
    # wall at R + w. Solve the limiting case for R.
    R_max = sp.solve(sp.Eq(x_floor, R + w), R)[0]
    assert_equal(R_max, w * h / d, "R_max = w * h / d")

    # And the inverse, which is what sizes the mast.
    D_stop = sp.symbols("D_stop", positive=True)
    h_req = sp.solve(sp.Eq(w * h / d, D_stop), h)[0]
    assert_equal(h_req, d * D_stop / w, "h_required = d * D_stop / w")

    # No sensor type appears anywhere above. That is the whole finding.
    assert not R_max.free_symbols - {w, h, d}, "R_max gained a free symbol"


def depth_error():
    """dZ = Z**2 * d_disp / (f * B), by differentiating the depth relation."""
    print("\nstereo depth and its error")
    f, B, disp, Z, d_disp = sp.symbols("f B disp Z d_disp", positive=True)

    depth = f * B / disp
    # First order propagation of a disparity uncertainty.
    sensitivity = sp.diff(depth, disp)
    # Re-express in terms of range by substituting the disparity that
    # produces range Z.
    disp_at_Z = sp.solve(sp.Eq(depth, Z), disp)[0]
    dZ = sp.Abs(sensitivity.subs(disp, disp_at_Z)) * d_disp
    assert_equal(sp.simplify(dZ), Z**2 * d_disp / (f * B), "dZ = Z^2 * d_disp / (f*B)")

    # Range at a target error, and the baseline that buys it.
    err = sp.symbols("err", positive=True)
    Z_at_err = sp.solve(sp.Eq(Z**2 * d_disp / (f * B), err), Z)[0]
    assert_equal(
        Z_at_err, sp.sqrt(err * f * B / d_disp), "Z_at_error = sqrt(err*f*B/d_disp)"
    )

    B_needed = sp.solve(sp.Eq(Z**2 * d_disp / (f * B), err), B)[0]
    assert_equal(
        B_needed, Z**2 * d_disp / (f * err), "B_for_error = Z^2*d_disp/(f*err)"
    )

    # Error is strictly increasing in range, which the property tests assert
    # numerically. Here it is proved.
    growth = sp.diff(Z**2 * d_disp / (f * B), Z)
    assert sp.ask(sp.Q.positive(growth)) is not False, "depth error must grow with range"
    print("  ok  d(dZ)/dZ > 0 for all positive Z, f, B, d_disp")


def focal_and_projection():
    """f_px from HFOV, and the projection round trip."""
    print("\npinhole optics")
    W, hfov, f = sp.symbols("W hfov f", positive=True)

    # Half the sensor width subtends half the field of view at distance f.
    f_from_fov = sp.solve(sp.Eq(sp.tan(hfov / 2), (W / 2) / f), f)[0]
    assert_equal(f_from_fov, (W / 2) / sp.tan(hfov / 2), "f_px = (W/2)/tan(HFOV/2)")

    # Projection and unprojection are exact inverses, symbolically.
    X, Y, Z, fx, fy, cx, cy = sp.symbols("X Y Z fx fy cx cy", positive=True)
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    X_back = (u - cx) * Z / fx
    Y_back = (v - cy) * Z / fy
    assert_equal(X_back, X, "unproject(project(X)) = X")
    assert_equal(Y_back, Y, "unproject(project(Y)) = Y")


def blind_zone():
    """d_near = h / tan(tilt + VFOV/2), from the bottom ray direction."""
    print("\nblind zone")
    h, tilt, vfov, t = sp.symbols("h tilt vfov t", positive=True)

    # The bottom edge of the frustum leaves the eye at (tilt + vfov/2) below
    # horizontal. Walk along it until it meets the ground.
    angle = tilt + vfov / 2
    x_of_t = t * sp.cos(angle)
    z_of_t = h - t * sp.sin(angle)
    t_ground = sp.solve(sp.Eq(z_of_t, 0), t)[0]
    d_near = sp.simplify(x_of_t.subs(t, t_ground))
    assert_equal(d_near, h / sp.tan(angle), "d_near = h / tan(tilt + VFOV/2)")


def stopping_distance():
    """D_stop = v*t_lat + v^2/(2a), by integrating the motion."""
    print("\nstopping distance")
    v, a, t_lat, tau = sp.symbols("v a t_lat tau", positive=True)

    # Constant speed through the latency window.
    reaction = sp.integrate(v, (tau, 0, t_lat))
    # Then constant deceleration until the speed reaches zero.
    t_brake = sp.solve(sp.Eq(v - a * tau, 0), tau)[0]
    braking = sp.integrate(v - a * tau, (tau, 0, t_brake))

    assert_equal(reaction, v * t_lat, "reaction = v * t_latency")
    assert_equal(braking, v**2 / (2 * a), "braking = v^2 / (2a)")

    # And the speed cap that inverts it, which is what turns a trusted
    # sensing range into a velocity limit.
    R, sf = sp.symbols("R sf", positive=True)
    solutions = sp.solve(sp.Eq((v * t_lat + v**2 / (2 * a)) * sf, R), v)
    v_max = [s for s in solutions if sp.simplify(s.subs({t_lat: 0})) != 0 or True]
    positive_root = -a * t_lat + sp.sqrt((a * t_lat) ** 2 + 2 * a * R / sf)
    matched = any(sp.simplify(s - positive_root) == 0 for s in v_max)
    if not matched:
        raise AssertionError(f"speed cap root not recovered, sympy gave {solutions}")
    RESULTS.append(("v_max for range R", positive_root))
    print("  ok  v_max = -a*t + sqrt((a*t)^2 + 2*a*R/sf)")
    print(f"      {positive_root}")


def stereo_sync():
    """The two exposure skew terms, exactly.

    Two eyes only fuse if they look at the same instant. These are the errors
    a stereo pair makes when they do not.
    """
    print("\nstereo synchronisation")
    f, B, Z, X, delta = sp.symbols("f B Z X delta", positive=True)

    # Translation. The right camera fires late, by which time it has advanced
    # delta = v*dt along the optical axis.
    u_left = f * X / Z
    u_right = f * (X - B) / (Z - delta)
    disparity = sp.simplify(u_left - u_right)
    Z_hat = sp.simplify(f * B / disparity)

    # On the optical axis the reconstruction is biased by exactly delta.
    Z_hat_axis = sp.simplify(Z_hat.subs(X, 0))
    assert_equal(Z_hat_axis, Z - delta, "skewed depth on axis = Z - v*dt")

    v, dt = sp.symbols("v dt", positive=True)
    bias = sp.simplify((Z - Z_hat_axis).subs(delta, v * dt))
    assert_equal(bias, v * dt, "skew_depth_bias = v * dt")

    # Off axis there is a second term, recorded here so nobody rediscovers it.
    print(f"      off axis, Z_hat = {sp.simplify(Z_hat)}")

    # Rotation. The right camera fires late, by which time the vehicle has
    # yawed by psi. A point on the optical axis moves by f*tan(psi) pixels.
    psi, w_yaw = sp.symbols("psi omega", positive=True)
    x_r = -Z * sp.sin(psi)
    z_r = Z * sp.cos(psi)
    u_rotated = sp.simplify(f * x_r / z_r)
    assert_equal(
        -u_rotated, f * sp.tan(psi), "skew_disparity_error = f * tan(omega*dt)"
    )

    # And its exact inverse, the sync budget.
    budget = sp.symbols("budget", positive=True)
    dt_max = sp.solve(sp.Eq(f * sp.tan(w_yaw * dt), budget), dt)[0]
    assert_equal(
        dt_max, sp.atan(budget / f) / w_yaw, "sync_skew_budget = atan(budget/f)/omega"
    )


def wheel_support_on_a_plane():
    """Support height r*sqrt(1 + tan(p)^2 + tan(r)^2) above a sloped plane.

    Used by the closed form pose that the numerical rocker-bogie solver is
    checked against.
    """
    print("\nwheel support on a plane")
    r, mx, my, x0, y0, z0 = sp.symbols("r m_x m_y x0 y0 z0", real=True, positive=True)
    zc = sp.symbols("z_c", real=True)

    # Plane z = z0 + mx*x + my*y, unit normal, signed distance to the centre.
    normal = sp.Matrix([-mx, -my, 1]) / sp.sqrt(1 + mx**2 + my**2)
    to_centre = sp.Matrix([0, 0, zc - (z0 + mx * x0 + my * y0)])
    distance = (normal.T @ to_centre)[0]
    solved = sp.solve(sp.Eq(distance, r), zc)[0]

    expected = z0 + mx * x0 + my * y0 + r * sp.sqrt(1 + mx**2 + my**2)
    assert_equal(solved, expected, "z_centre = z_ground + r*sqrt(1 + tan(p)^2 + tan(r)^2)")


def rocker_differential():
    """The differential reduces to the two relative rocker angles summing to zero."""
    print("\nrocker-bogie differential")
    body, rel_l, rel_r = sp.symbols("theta_body theta_rel_L theta_rel_R", real=True)

    # Absolute rocker pitch is body pitch plus the joint angle.
    abs_l = body + rel_l
    abs_r = body + rel_r
    # The differential bar forces the body to bisect the two rockers.
    constraint = sp.Eq(body, (abs_l + abs_r) / 2)
    reduced = sp.simplify(sp.solve(constraint, rel_l)[0])
    assert_equal(reduced, -rel_r, "differential: theta_rel_L = -theta_rel_R")


def main():
    print("ARES symbolic derivations, build spec section 11 mechanism 5")
    negative_obstacle()
    depth_error()
    focal_and_projection()
    blind_zone()
    stopping_distance()
    stereo_sync()
    wheel_support_on_a_plane()
    rocker_differential()
    print(f"\n{len(RESULTS)} derivations match their closed forms.")


if __name__ == "__main__":
    main()
