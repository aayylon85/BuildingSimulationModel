"""
Implements interior surface-to-surface longwave radiation exchange.

This module provides multiple methods for calculating radiation exchange between
interior surfaces:

1. AreaWeightedMRT (default): Simple area-weighted Mean Radiant Temperature
2. CarrollMRT: Iterative MRT that preserves T^4 nonlinearity (EnergyPlus-style)
3. ViewFactorMethod: Full view factor matrix for rectangular enclosures

References:
    - EnergyPlus HeatBalanceIntRadExchange.cc (Hottel's ScriptF method)
    - Carroll, J.A. 1980. "An MRT Method of Computing Radiant Energy Exchange
      in Rooms." Proceedings, 2nd BSME Conference, Houston, TX.
    - ASHRAE Fundamentals 2021, Chapter 4 "Heat Transfer"
    - Hottel & Sarofim, "Radiative Transfer", McGraw-Hill, 1967
"""

import math
from bsm.core.constants import STEFAN_BOLTZMANN


class InteriorLongwaveRadiation:
    """
    Factory class for interior radiation methods.

    Supports three methods:
    - 'area_weighted': Simple area-weighted MRT (fastest, least accurate for gradients)
    - 'carroll': Carroll MRT method preserving T^4 nonlinearity (good balance)
    - 'view_factor': Full view factor matrix (most accurate, requires geometry)
    """

    METHODS = ['area_weighted', 'carroll', 'view_factor']

    def __init__(self, emissivity=0.9, method='area_weighted', zone_geometry=None):
        """
        Initialize the interior radiation calculator.

        Args:
            emissivity (float): Surface emissivity (default 0.9 per ASHRAE 140).
            method (str): Radiation method - 'area_weighted', 'carroll', or 'view_factor'
            zone_geometry (dict): Zone dimensions for view factor calculation.
                                  Required only for 'view_factor' method.
                                  Format: {'length': m, 'width': m, 'height': m}
        """
        if not 0.0 <= emissivity <= 1.0:
            raise ValueError("Emissivity must be between 0 and 1.")
        if method not in self.METHODS:
            raise ValueError(f"Method must be one of {self.METHODS}")

        self.emissivity = emissivity
        self.method = method
        self.zone_geometry = zone_geometry

        # Initialize view factors if needed
        if method == 'view_factor':
            if zone_geometry is None:
                raise ValueError("zone_geometry required for view_factor method")
            self._view_factors = self._calculate_view_factors(zone_geometry)
        else:
            self._view_factors = None

    def calculate_mrt_and_coefficient(self, surface_temps, surface_areas,
                                       current_surface_name):
        """
        Calculate MRT and linearized radiation coefficient for a surface.

        Dispatches to appropriate method based on initialization.
        """
        if self.method == 'area_weighted':
            return self._area_weighted_mrt(surface_temps, surface_areas,
                                           current_surface_name)
        elif self.method == 'carroll':
            return self._carroll_mrt(surface_temps, surface_areas,
                                     current_surface_name)
        elif self.method == 'view_factor':
            return self._view_factor_mrt(surface_temps, surface_areas,
                                         current_surface_name)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _area_weighted_mrt(self, surface_temps, surface_areas, current_surface_name):
        """
        Area-weighted MRT method (original implementation).

        Uses area-weighted view factor approximation:
            F_ij ≈ A_j / (Total_Area - A_i)
        """
        if current_surface_name not in surface_temps:
            raise ValueError(f"Surface '{current_surface_name}' not found in surface_temps")

        T_surf = surface_temps[current_surface_name]

        # Calculate area-weighted MRT from other surfaces
        total_other_area = 0.0
        weighted_temp_sum = 0.0

        for name, T in surface_temps.items():
            if name != current_surface_name:
                A = surface_areas.get(name, 0.0)
                total_other_area += A
                weighted_temp_sum += A * T

        if total_other_area > 0:
            T_MRT = weighted_temp_sum / total_other_area
        else:
            T_MRT = T_surf

        h_r = self._calculate_h_r(T_surf, T_MRT)
        return T_MRT, h_r

    def _carroll_mrt(self, surface_temps, surface_areas, current_surface_name):
        """
        Carroll MRT method that preserves T^4 nonlinearity.

        This method calculates MRT such that the linearized radiation
        exchange properly represents the T^4 physics. Uses Fp (grey-body
        view factor) weighting.

        Reference: EnergyPlus HeatBalanceIntRadExchange.cc CarrollMRT
        """
        if current_surface_name not in surface_temps:
            raise ValueError(f"Surface '{current_surface_name}' not found")

        T_surf = surface_temps[current_surface_name]
        T_surf_K = T_surf + 273.15
        A_surf = surface_areas.get(current_surface_name, 1.0)

        # Calculate Fp factors (grey-body view factors with emissivity)
        # Fp_i = ε_i * A_i / Σ(ε_j * A_j)
        # Simplifies to: Fp_i = A_i / Σ(A_j) when all ε are equal
        total_eps_area = sum(self.emissivity * A for A in surface_areas.values())

        if total_eps_area <= 0:
            return T_surf, self._calculate_h_r(T_surf, T_surf)

        # Calculate MRT using T^4 weighting (Carroll method)
        # T_MRT^4 = Σ(Fp_j * T_j^4) for all j ≠ i
        sum_fp_t4 = 0.0
        sum_fp = 0.0

        for name, T in surface_temps.items():
            if name != current_surface_name:
                A = surface_areas.get(name, 0.0)
                Fp = self.emissivity * A / total_eps_area
                T_K = T + 273.15
                sum_fp_t4 += Fp * (T_K ** 4)
                sum_fp += Fp

        if sum_fp > 0:
            # MRT from T^4 average, then convert back
            T_MRT_K4 = sum_fp_t4 / sum_fp
            T_MRT_K = T_MRT_K4 ** 0.25
            T_MRT = T_MRT_K - 273.15
        else:
            T_MRT = T_surf

        # Calculate linearized h_r at the actual MRT
        # This preserves the T^4 nonlinearity in the coefficient
        h_r = self._calculate_h_r(T_surf, T_MRT)

        return T_MRT, h_r

    def _view_factor_mrt(self, surface_temps, surface_areas, current_surface_name):
        """
        View factor method using pre-calculated geometric view factors.

        Uses exact geometric view factors for rectangular enclosure,
        then applies grey-body ScriptF-style calculation.
        """
        if current_surface_name not in surface_temps:
            raise ValueError(f"Surface '{current_surface_name}' not found")
        if self._view_factors is None:
            raise ValueError("View factors not calculated - check zone_geometry")

        T_surf = surface_temps[current_surface_name]
        T_surf_K = T_surf + 273.15

        # Get view factors from this surface to all others
        F_to_others = self._view_factors.get(current_surface_name, {})

        # Calculate ScriptF-weighted T^4 MRT
        # ScriptF accounts for grey-body interchange between surfaces
        sum_scriptf_t4 = 0.0
        sum_scriptf = 0.0

        for name, T in surface_temps.items():
            if name != current_surface_name:
                F_ij = F_to_others.get(name, 0.0)
                # Grey-body ScriptF: F_ij * ε_j / (1 - (1-ε_i)*(1-ε_j)*F_ij*F_ji)
                # Simplified for uniform emissivity:
                ScriptF = F_ij * self.emissivity

                T_K = T + 273.15
                sum_scriptf_t4 += ScriptF * (T_K ** 4)
                sum_scriptf += ScriptF

        if sum_scriptf > 0:
            T_MRT_K4 = sum_scriptf_t4 / sum_scriptf
            T_MRT_K = T_MRT_K4 ** 0.25
            T_MRT = T_MRT_K - 273.15
        else:
            T_MRT = T_surf

        h_r = self._calculate_h_r(T_surf, T_MRT)
        return T_MRT, h_r

    def _calculate_h_r(self, T_surf_c, T_MRT_c):
        """
        Calculate linearized radiation coefficient h_r.

        h_r = ε × σ × (T_surf + T_MRT) × (T_surf² + T_MRT²)
        """
        T_surf_K = T_surf_c + 273.15
        T_MRT_K = T_MRT_c + 273.15

        if abs(T_MRT_K - T_surf_K) < 0.01:
            # Use derivative form for small differences
            h_r = 4.0 * self.emissivity * STEFAN_BOLTZMANN * (T_surf_K ** 3)
        else:
            h_r = (self.emissivity * STEFAN_BOLTZMANN *
                   (T_surf_K + T_MRT_K) * (T_surf_K**2 + T_MRT_K**2))
        return h_r

    def _calculate_view_factors(self, geometry):
        """
        Calculate view factors for a rectangular enclosure.

        Uses analytical formulas for parallel and perpendicular rectangles.

        Args:
            geometry: dict with 'length', 'width', 'height' in meters

        Returns:
            dict: Nested dict of view factors F[surface_i][surface_j]
        """
        L = geometry['length']
        W = geometry['width']
        H = geometry['height']

        # Surface names matching BESTEST convention
        surfaces = ['floor', 'roof', 'south_wall', 'north_wall', 'east_wall', 'west_wall']

        # Surface areas
        A_floor = L * W
        A_roof = L * W
        A_south = L * H
        A_north = L * H
        A_east = W * H
        A_west = W * H

        areas = {
            'floor': A_floor, 'roof': A_roof,
            'south_wall': A_south, 'north_wall': A_north,
            'east_wall': A_east, 'west_wall': A_west
        }

        # Calculate view factors using analytical formulas
        F = {s: {} for s in surfaces}

        # Floor-to-Roof (parallel rectangles, distance H)
        F_floor_roof = self._view_factor_parallel_rectangles(L, W, H)
        F['floor']['roof'] = F_floor_roof
        F['roof']['floor'] = F_floor_roof  # By symmetry

        # Floor-to-Walls (perpendicular rectangles)
        # Floor to South/North walls (L x W floor to L x H wall)
        F_floor_south = self._view_factor_perpendicular_rectangles(L, W, H)
        F['floor']['south_wall'] = F_floor_south
        F['floor']['north_wall'] = F_floor_south
        F['roof']['south_wall'] = F_floor_south
        F['roof']['north_wall'] = F_floor_south

        # Floor to East/West walls (L x W floor to W x H wall)
        F_floor_east = self._view_factor_perpendicular_rectangles(W, L, H)
        F['floor']['east_wall'] = F_floor_east
        F['floor']['west_wall'] = F_floor_east
        F['roof']['east_wall'] = F_floor_east
        F['roof']['west_wall'] = F_floor_east

        # South-to-North walls (parallel rectangles, distance W)
        F_south_north = self._view_factor_parallel_rectangles(L, H, W)
        F['south_wall']['north_wall'] = F_south_north
        F['north_wall']['south_wall'] = F_south_north

        # East-to-West walls (parallel rectangles, distance L)
        F_east_west = self._view_factor_parallel_rectangles(W, H, L)
        F['east_wall']['west_wall'] = F_east_west
        F['west_wall']['east_wall'] = F_east_west

        # Wall-to-Wall perpendicular (adjacent walls)
        # South/North to East/West
        F_south_east = self._view_factor_perpendicular_rectangles(H, L, W)
        F['south_wall']['east_wall'] = F_south_east
        F['south_wall']['west_wall'] = F_south_east
        F['north_wall']['east_wall'] = F_south_east
        F['north_wall']['west_wall'] = F_south_east

        # Use reciprocity for reverse directions: A_i * F_ij = A_j * F_ji
        F['east_wall']['south_wall'] = F_south_east * A_south / A_east
        F['east_wall']['north_wall'] = F_south_east * A_north / A_east
        F['west_wall']['south_wall'] = F_south_east * A_south / A_west
        F['west_wall']['north_wall'] = F_south_east * A_north / A_west

        # Wall to Floor/Roof (use reciprocity)
        F['south_wall']['floor'] = F_floor_south * A_floor / A_south
        F['south_wall']['roof'] = F_floor_south * A_roof / A_south
        F['north_wall']['floor'] = F_floor_south * A_floor / A_north
        F['north_wall']['roof'] = F_floor_south * A_roof / A_north
        F['east_wall']['floor'] = F_floor_east * A_floor / A_east
        F['east_wall']['roof'] = F_floor_east * A_roof / A_east
        F['west_wall']['floor'] = F_floor_east * A_floor / A_west
        F['west_wall']['roof'] = F_floor_east * A_roof / A_west

        # Self view factors are 0 for flat surfaces
        for s in surfaces:
            F[s][s] = 0.0

        return F

    def _view_factor_parallel_rectangles(self, a, b, c):
        """
        View factor between two identical parallel rectangles.

        From Howell's view factor catalog, Configuration C-13.

        Args:
            a: Length of rectangles
            b: Width of rectangles
            c: Distance between rectangles

        Returns:
            View factor F_12
        """
        X = a / c
        Y = b / c

        # Avoid log(0) for edge cases
        if X < 1e-10 or Y < 1e-10:
            return 0.0

        term1 = math.log(
            (1 + X**2) * (1 + Y**2) / (1 + X**2 + Y**2)
        )

        X_term = X * math.sqrt(1 + Y**2) * math.atan(X / math.sqrt(1 + Y**2))
        Y_term = Y * math.sqrt(1 + X**2) * math.atan(Y / math.sqrt(1 + X**2))

        F = (2 / (math.pi * X * Y)) * (
            0.5 * term1 + X_term + Y_term - X * math.atan(X) - Y * math.atan(Y)
        )

        return max(0.0, min(1.0, F))

    def _view_factor_perpendicular_rectangles(self, a, b, c):
        """
        View factor from rectangle 1 to perpendicular rectangle 2 sharing common edge.

        Rectangle 1: a x b (a is common edge)
        Rectangle 2: a x c (a is common edge)

        From Howell's view factor catalog, Configuration C-14.
        """
        H = b / a if a > 0 else 0
        W = c / a if a > 0 else 0

        if H < 1e-10 or W < 1e-10:
            return 0.0

        A = (1 + H**2) * (1 + W**2) / (1 + H**2 + W**2)
        B = ((W**2 * (1 + H**2 + W**2)) /
             ((1 + W**2) * (H**2 + W**2)))
        C = ((H**2 * (1 + H**2 + W**2)) /
             ((1 + H**2) * (H**2 + W**2)))

        term1 = W * math.atan(1/W) + H * math.atan(1/H)
        term2 = math.sqrt(H**2 + W**2) * math.atan(1/math.sqrt(H**2 + W**2))
        term3 = 0.25 * math.log(A * (B ** (W**2)) * (C ** (H**2)))

        F = (1 / (math.pi * H)) * (term1 - term2 + term3)

        return max(0.0, min(1.0, F))

    def calculate_net_radiation_flux(self, T_surface_c, T_MRT_c, h_r):
        """
        Calculate net radiation heat flux to a surface.

        Positive flux means heat gain to the surface.
        Negative flux means heat loss from the surface.
        """
        return h_r * (T_MRT_c - T_surface_c)


# Convenience aliases for backward compatibility
AreaWeightedMRT = lambda emissivity=0.9: InteriorLongwaveRadiation(emissivity, method='area_weighted')
CarrollMRT = lambda emissivity=0.9: InteriorLongwaveRadiation(emissivity, method='carroll')


if __name__ == '__main__':
    # Example: Compare methods for BESTEST zone
    print("=" * 70)
    print("Interior Longwave Radiation Method Comparison")
    print("BESTEST Zone: 8m x 6m x 2.7m")
    print("=" * 70)
    print()

    # Define zone surfaces (BESTEST Case 900 - high mass, heated floor)
    surface_temps = {
        'floor': 28.0,       # Heated by solar - warmer
        'roof': 23.0,        # Near zone air temp
        'south_wall': 24.0,  # Some solar absorption
        'north_wall': 22.5,  # Shaded wall
        'east_wall': 23.5,
        'west_wall': 23.5,
    }

    surface_areas = {
        'floor': 8.0 * 6.0,           # 48 m²
        'roof': 8.0 * 6.0,            # 48 m²
        'south_wall': 8.0 * 2.7,      # 21.6 m²
        'north_wall': 8.0 * 2.7,      # 21.6 m²
        'east_wall': 6.0 * 2.7,       # 16.2 m²
        'west_wall': 6.0 * 2.7,       # 16.2 m²
    }

    zone_geometry = {'length': 8.0, 'width': 6.0, 'height': 2.7}

    print("Surface Temperatures (simulating solar-heated high-mass floor):")
    for name in surface_temps:
        print(f"  {name}: T={surface_temps[name]:.1f}°C, A={surface_areas[name]:.1f} m²")
    print()

    # Test each method
    methods = [
        ('Area-Weighted MRT', 'area_weighted', None),
        ('Carroll MRT', 'carroll', None),
        ('View Factor', 'view_factor', zone_geometry),
    ]

    print("-" * 70)
    print(f"{'Method':<20} {'MRT (°C)':<12} {'h_r (W/m²K)':<14} {'q_rad (W/m²)':<14} {'Q_total (W)':<12}")
    print("-" * 70)

    for method_name, method_id, geom in methods:
        rad_calc = InteriorLongwaveRadiation(
            emissivity=0.9,
            method=method_id,
            zone_geometry=geom
        )

        T_MRT, h_r = rad_calc.calculate_mrt_and_coefficient(
            surface_temps, surface_areas, 'floor'
        )

        q_rad = rad_calc.calculate_net_radiation_flux(
            surface_temps['floor'], T_MRT, h_r
        )
        Q_total = q_rad * surface_areas['floor']

        print(f"{method_name:<20} {T_MRT:<12.2f} {h_r:<14.2f} {q_rad:<14.2f} {Q_total:<12.1f}")

    print("-" * 70)
    print()
    print("Notes:")
    print("  - Negative q_rad means floor is losing heat to cooler surroundings")
    print("  - Carroll MRT preserves T^4 nonlinearity, giving lower MRT")
    print("  - View Factor uses exact geometric view factors for rectangular rooms")
    print("  - Higher |Q_total| means more heat removed from thermal mass")
