"""
BESTEST Case Library - Loads and manages ASHRAE Standard 140 test case definitions.

This module provides functionality to load test case specifications from JSON files,
resolve inheritance chains, and return complete case configurations.
"""
import json
import os
from typing import Dict, List, Optional
from copy import deepcopy


class BESTESTCase:
    """Represents a single BESTEST test case with all its specifications."""

    def __init__(self, case_id: str, case_data: dict):
        """
        Initialize a BESTEST case.

        Args:
            case_id: Case identifier (e.g., "600", "610")
            case_data: Complete case specification dictionary
        """
        self.case_id = case_id
        self.data = case_data

    def __getitem__(self, key):
        """Allow dictionary-like access to case data."""
        return self.data[key]

    def get(self, key, default=None):
        """Get a value from case data with a default."""
        return self.data.get(key, default)

    def to_dict(self) -> dict:
        """Return the complete case data as a dictionary."""
        return self.data


class BESTESTCaseLibrary:
    """
    Manages loading and access to BESTEST test case definitions.

    This class handles JSON parsing, inheritance resolution, and provides
    methods to access individual cases or lists of cases.
    """

    def __init__(self, cases_file: str):
        """
        Initialize the case library.

        Args:
            cases_file: Path to the JSON file containing test case definitions
        """
        self.cases_file = cases_file
        self.base_case = None
        self.test_cases = {}
        self._load_cases()

    def _load_cases(self):
        """Load and parse the test cases JSON file."""
        if not os.path.exists(self.cases_file):
            raise FileNotFoundError(f"Test cases file not found: {self.cases_file}")

        with open(self.cases_file, 'r') as f:
            data = json.load(f)

        # Load base case
        self.base_case = data.get('base_case')
        if not self.base_case:
            raise ValueError("Test cases file must contain a 'base_case' definition")

        # Load test cases
        test_case_list = data.get('test_cases', [])
        for case_spec in test_case_list:
            case_id = case_spec.get('case_id')
            if not case_id:
                raise ValueError("Each test case must have a 'case_id'")

            # Resolve inheritance and create complete case
            complete_case = self._resolve_case(case_spec)
            self.test_cases[case_id] = BESTESTCase(case_id, complete_case)

    def _resolve_case(self, case_spec: dict) -> dict:
        """
        Resolve inheritance and merge modifications to create a complete case.

        Args:
            case_spec: Test case specification with potential 'inherits_from' and 'modifications'

        Returns:
            Complete case dictionary with all inherited and modified values
        """
        inherits_from = case_spec.get('inherits_from')

        if inherits_from == 'base_case':
            # Start with a deep copy of the base case
            complete_case = deepcopy(self.base_case)
        elif inherits_from:
            # Inherit from another test case
            if inherits_from not in self.test_cases:
                raise ValueError(f"Case '{case_spec.get('case_id')}' inherits from unknown case '{inherits_from}'")
            complete_case = deepcopy(self.test_cases[inherits_from].data)
        else:
            # No inheritance, start with empty dict
            complete_case = {}

        # Override case_id and description
        complete_case['case_id'] = case_spec.get('case_id')
        complete_case['description'] = case_spec.get('description', complete_case.get('description', ''))

        # Apply modifications if present
        modifications = case_spec.get('modifications', {})
        if modifications:
            complete_case = self._apply_modifications(complete_case, modifications)

        return complete_case

    def _apply_modifications(self, base_data: dict, modifications: dict) -> dict:
        """
        Apply modifications to base data using deep merge.

        Args:
            base_data: Base case data
            modifications: Modifications to apply

        Returns:
            Modified case data
        """
        result = deepcopy(base_data)

        for key, value in modifications.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                # Deep merge for nested dictionaries
                result[key] = self._deep_merge(result[key], value)
            elif isinstance(value, list) and key == 'remove_windows':
                # Special handling for removing windows
                if 'windows' in result:
                    result['windows'] = [w for w in result['windows'] if w.get('wall_name') not in value]
            else:
                # Direct replacement for other types
                result[key] = deepcopy(value)

        return result

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """
        Deep merge two dictionaries.

        Args:
            base: Base dictionary
            override: Dictionary with values to override

        Returns:
            Merged dictionary
        """
        result = deepcopy(base)

        for key, value in override.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)

        return result

    def get_case(self, case_id: str) -> BESTESTCase:
        """
        Get a specific test case by ID.

        Args:
            case_id: Case identifier (e.g., "600")

        Returns:
            BESTESTCase object

        Raises:
            KeyError: If case_id is not found
        """
        if case_id not in self.test_cases:
            raise KeyError(f"Test case '{case_id}' not found. Available cases: {self.list_cases()}")
        return self.test_cases[case_id]

    def list_cases(self) -> List[str]:
        """
        Get a list of all available test case IDs.

        Returns:
            List of case IDs
        """
        return sorted(self.test_cases.keys())

    def get_all_cases(self) -> Dict[str, BESTESTCase]:
        """
        Get all test cases.

        Returns:
            Dictionary mapping case IDs to BESTESTCase objects
        """
        return self.test_cases

    def __len__(self):
        """Return the number of test cases."""
        return len(self.test_cases)

    def __contains__(self, case_id):
        """Check if a case ID exists."""
        return case_id in self.test_cases


def load_section_5_2_cases() -> BESTESTCaseLibrary:
    """
    Convenience function to load Section 5.2 test cases.

    Returns:
        BESTESTCaseLibrary for Section 5.2

    Raises:
        FileNotFoundError: If the Section 5.2 cases file is not found
    """
    # Get the directory containing this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cases_file = os.path.join(current_dir, 'section_5_2_cases.json')

    return BESTESTCaseLibrary(cases_file)


if __name__ == '__main__':
    # Test the case library
    try:
        library = load_section_5_2_cases()
        print(f"Loaded {len(library)} test cases:")
        for case_id in library.list_cases():
            case = library.get_case(case_id)
            print(f"  Case {case_id}: {case['description']}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please create section_5_2_cases.json first.")
