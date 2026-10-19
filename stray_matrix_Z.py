
    # 5. SMU별 Test 성능
    print("\n" + "=" * 68)
    print("SAME-SMU PERFORMANCE BY GROUP (TEST ONLY)")
    print("=" * 68)

    for g in range(6):
        idx = test[SMU[i[test]] == g]

        print_metrics(
            f"SMU {g+1} ({len(idx)} test pairs)",
            actual, predictions, idx
        )

    # 6. Pin 거리별 성능
    print("\n" + "=" * 68)
    print("PAIR DISTANCE PERFORMANCE (TEST ONLY)")
    print("=" * 68)

    categories = [
        "Adjacent Even",
        "Adjacent Odd",
        "Distance 2",
        "Distance >=3"
    ]

    for category in categories:
        idx = np.array(
            [k for k in test
             if pair_type(i[k], j[k]) == category],
            dtype=int
        )

        print_metrics(
            f"{category} ({len(idx)} pairs)",
            actual, predictions, idx
        )

    # 7. Probe Card 근접 효과를 배제한 비교
    # 동일 SMU + Pin 거리 >= 3
    far_test = np.array(
        [k for k in test
         if abs(i[k] - j[k]) >= 3],
        dtype=int
    )

    print_metrics(
        "SAME-SMU DISTANCE >= 3 (TEST)",
        actual, predictions, far_test
    )

    # 8. SMU별 피팅 파라미터
    print("\n" + "=" * 68)
    print("FITTED PARASITIC PARAMETERS")
    print("=" * 68)

    print(
        f"{'SMU':<6}"
        f"{'d_real (MOhm)':>18}"
        f"{'d_imag (MOhm)':>18}"
        f"{'y_real (uS)':>18}"
        f"{'y_imag (uS)':>18}"
    )

    for g in range(6):
        print(
            f"{g+1:<6}"
            f"{d[g].real/1e6:18.6f}"
            f"{d[g].imag/1e6:18.6f}"
            f"{y[g].real*1e6:18.6f}"
            f"{y[g].imag*1e6:18.6f}"
        )

    # 9. Pin별 직렬 임피던스 범위
    print("\nPIN SERIES IMPEDANCE SUMMARY")
    print(
        "Mean |z_pin| (MOhm):",
        np.mean(np.abs(z_pin))/1e6
    )
    print(
        "Min |z_pin| (MOhm):",
        np.min(np.abs(z_pin))/1e6
    )
    print(
        "Max |z_pin| (MOhm):",
        np.max(np.abs(z_pin))/1e6
    )

    # 10. 결과 반환
    return {
        "z_pin": z_pin,
        "series_parasitic": d,
        "parallel_parasitic": y,
        "actual": actual,
        "pred_base": pred_base,
        "pred_z": pred_z,
        "pred_y": pred_y,
        "train_indices": train,
        "test_indices": test,
        "pair_i": i,
        "pair_j": j
    }


# ============================================================
# EXECUTION
# ============================================================

results = run_comparison(Z)
