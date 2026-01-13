"""
Angular-dependent window transmittance models for ASHRAE 140 compliance.

Implements ASHRAE 140-2023 Table 7-12 angular transmittance function using
polynomial fits to the tabulated data.
"""
import numpy as np
import math


class AngularTransmittanceModel:
    """
    ASHRAE 140-2023 Table 7-12 angular transmittance model.

    Implements angular-dependent SHGC using a polynomial fit to tabulated data.
    The model uses the angular profile from Table 7-12 and scales it to match
    the user-specified normal-incidence SHGC value.
    """

    # Table 7-12 data from ASHRAE 140-2023 (Documentation/Table7-12.csv)
    # Angle (degrees), SHGC
    TABLE_7_12_DATA = np.array([
        [0, 0.769],
        [10, 0.768],
        [20, 0.766],
        [30, 0.761],
        [40, 0.748],
        [50, 0.718],
        [60, 0.651],
        [70, 0.509],
        [80, 0.267],
        [90, 0.000]
    ])

    # Hemispherical average from Table 7-12
    TABLE_7_12_HEMISPHERICAL = 0.670

    def __init__(self, normal_transmittance=0.789):
        """
        Initialize angular transmittance model.

        Args:
            normal_transmittance: SHGC at normal incidence (default: 0.789 for ASHRAE 140 Case 600)
        """
        self.shgc_normal = normal_transmittance

        # Reference normal SHGC from Table 7-12
        self.shgc_normal_table = self.TABLE_7_12_DATA[0, 1]  # 0.769

        # Fit polynomial to Table 7-12 data
        self._fit_polynomial()

    def _fit_polynomial(self):
        """
        Fit 4th-order polynomial to ASHRAE 140 Table 7-12 data.

        Fits polynomial to the normalized SHGC ratio: SHGC(θ)/SHGC(0°)
        This allows scaling to any normal-incidence SHGC value.
        """
        angles_deg = self.TABLE_7_12_DATA[:, 0]
        shgc_values = self.TABLE_7_12_DATA[:, 1]

        # Normalize by SHGC at normal incidence
        shgc_ratios = shgc_values / self.shgc_normal_table

        # Fit 4th-order polynomial: ratio = c0 + c1*θ + c2*θ² + c3*θ³ + c4*θ⁴
        # Use angles in degrees (not radians) for better numerical conditioning
        self.poly_coeffs = np.polyfit(angles_deg, shgc_ratios, deg=4)

        # Verify fit quality at data points
        max_error = 0.0
        for angle_deg, shgc_ratio_actual in zip(angles_deg, shgc_ratios):
            shgc_ratio_fit = np.polyval(self.poly_coeffs, angle_deg)
            error = abs(shgc_ratio_fit - shgc_ratio_actual)
            max_error = max(max_error, error)

        if max_error > 0.01:  # Warn if fit error > 1%
            print(f"Warning: Polynomial fit max error = {max_error:.4f} ({max_error*100:.2f}%)")

    def calculate_transmittance(self, incident_angle_deg):
        """
        Calculate SHGC for given incident angle.

        Args:
            incident_angle_deg: Angle between sun ray and window normal (degrees, 0-90)

        Returns:
            shgc: Solar Heat Gain Coefficient at this angle (0-1)
        """
        # Handle edge cases
        if incident_angle_deg >= 90.0:
            return 0.0
        if incident_angle_deg < 0.0:
            incident_angle_deg = 0.0

        # Evaluate polynomial to get normalized ratio
        shgc_ratio = np.polyval(self.poly_coeffs, incident_angle_deg)

        # Scale by user-specified normal SHGC
        shgc = self.shgc_normal * shgc_ratio

        # Ensure result is in valid range [0, shgc_normal]
        return max(0.0, min(self.shgc_normal, shgc))

    def calculate_hemispherical_average_transmittance(self):
        """
        Calculate hemispherical-averaged transmittance for diffuse radiation.

        Uses the hemispherical average from Table 7-12, scaled to match
        the user-specified normal SHGC.

        Returns:
            shgc_hemispherical: Average SHGC over hemisphere
        """
        # Ratio of hemispherical to normal from Table 7-12
        ratio = self.TABLE_7_12_HEMISPHERICAL / self.shgc_normal_table  # 0.670 / 0.769 = 0.871

        # Scale by user-specified normal SHGC
        return self.shgc_normal * ratio

    def get_fit_info(self):
        """
        Get information about the polynomial fit for validation purposes.

        Returns:
            dict: Information about the fit including coefficients and errors
        """
        angles_deg = self.TABLE_7_12_DATA[:, 0]
        shgc_values = self.TABLE_7_12_DATA[:, 1]
        shgc_ratios = shgc_values / self.shgc_normal_table

        errors = []
        for angle_deg, shgc_ratio_actual in zip(angles_deg, shgc_ratios):
            shgc_ratio_fit = np.polyval(self.poly_coeffs, angle_deg)
            error = abs(shgc_ratio_fit - shgc_ratio_actual)
            errors.append(error)

        return {
            'polynomial_order': len(self.poly_coeffs) - 1,
            'coefficients': self.poly_coeffs.tolist(),
            'max_error': max(errors),
            'mean_error': np.mean(errors),
            'shgc_normal_table': self.shgc_normal_table,
            'shgc_normal_user': self.shgc_normal,
            'hemispherical_avg': self.calculate_hemispherical_average_transmittance()
        }


if __name__ == '__main__':
    # Test the angular transmittance model
    print("ASHRAE 140-2023 Table 7-12 Angular Transmittance Model")
    print("=" * 60)

    # Create model with BESTEST Case 600 normal SHGC
    model = AngularTransmittanceModel(normal_transmittance=0.789)

    # Print fit information
    fit_info = model.get_fit_info()
    print(f"\nPolynomial Order: {fit_info['polynomial_order']}")
    print(f"Table 7-12 Normal SHGC: {fit_info['shgc_normal_table']:.3f}")
    print(f"User-Specified Normal SHGC: {fit_info['shgc_normal_user']:.3f}")
    print(f"Hemispherical Average: {fit_info['hemispherical_avg']:.3f}")
    print(f"Max Fit Error: {fit_info['max_error']:.4f} ({fit_info['max_error']*100:.2f}%)")
    print(f"Mean Fit Error: {fit_info['mean_error']:.4f} ({fit_info['mean_error']*100:.2f}%)")

    # Test at Table 7-12 data points
    print("\nValidation at Table 7-12 Data Points:")
    print(f"{'Angle':<8} {'Table 7-12':<12} {'Model':<12} {'Error':<12}")
    print("-" * 48)

    for angle, shgc_table in model.TABLE_7_12_DATA:
        # Scale table value to match user's normal SHGC
        shgc_table_scaled = shgc_table * (model.shgc_normal / model.shgc_normal_table)
        shgc_model = model.calculate_transmittance(angle)
        error = abs(shgc_model - shgc_table_scaled)
        print(f"{angle:6.0f}°  {shgc_table_scaled:10.4f}   {shgc_model:10.4f}   {error:10.4f}")

    # Test at intermediate angles
    print("\nInterpolation at Intermediate Angles:")
    print(f"{'Angle':<8} {'SHGC':<12}")
    print("-" * 24)
    for angle in [0, 15, 25, 35, 45, 55, 65, 75, 85, 90]:
        shgc = model.calculate_transmittance(angle)
        print(f"{angle:6.0f}°  {shgc:10.4f}")
