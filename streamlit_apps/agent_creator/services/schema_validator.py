"""Schema validation for scratch.json."""

from typing import Tuple, List, Dict, Any
from pydantic import ValidationError

from ..schemas.scratch_schema import ScratchSchema


class SchemaValidator:
    """Validate generated scratch.json against the schema."""

    def validate(self, json_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate JSON against ScratchSchema.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        try:
            ScratchSchema(**json_data)
            return True, []
        except ValidationError as e:
            errors = [f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()]
            return False, errors

    def fix_common_issues(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to auto-fix common LLM generation issues.

        Returns:
            Fixed JSON data (copy, does not modify original)
        """
        fixed = json_data.copy()

        # Ensure state fields have correct defaults
        fixed["daily_plan"] = None
        fixed["f_daily_schedule"] = []
        fixed["just_arrived"] = False

        # Ensure importance triggers match reflection_threshold
        if "reflection_threshold" in fixed:
            fixed["importance_trigger_max"] = fixed["reflection_threshold"]
            fixed["importance_trigger_curr"] = fixed["reflection_threshold"]

        # Ensure default cognitive parameters if missing
        if "perception_bandwidth" not in fixed:
            fixed["perception_bandwidth"] = 8
        if "reflection_threshold" not in fixed:
            fixed["reflection_threshold"] = 150
            fixed["importance_trigger_max"] = 150
            fixed["importance_trigger_curr"] = 150
        if "recency_w" not in fixed:
            fixed["recency_w"] = 1.0
        if "relevance_w" not in fixed:
            fixed["relevance_w"] = 1.0
        if "importance_w" not in fixed:
            fixed["importance_w"] = 1.0
        if "recency_decay" not in fixed:
            fixed["recency_decay"] = 0.995

        # Ensure relationship_models exists
        if "relationship_models" not in fixed:
            fixed["relationship_models"] = {}

        # Normalize thermal_preference
        thermal_map = {
            "cool": "cool",
            "cold": "cool",
            "neutral": "neutral",
            "normal": "neutral",
            "slightly warm": "slightly_warm",
            "slightly_warm": "slightly_warm",
            "warm": "warm",
            "hot": "warm"
        }
        if "thermal_preference" in fixed:
            val = fixed["thermal_preference"].lower().strip()
            fixed["thermal_preference"] = thermal_map.get(val, "neutral")

        # Normalize noise_sensitivity
        noise_map = {
            "low": "low",
            "none": "low",
            "moderate": "moderate",
            "medium": "moderate",
            "high": "high",
            "sensitive": "high"
        }
        if "noise_sensitivity" in fixed:
            val = fixed["noise_sensitivity"].lower().strip()
            fixed["noise_sensitivity"] = noise_map.get(val, "moderate")

        # Normalize light_preference
        light_map = {
            "natural": "natural",
            "daylight": "natural",
            "task": "task",
            "focused": "task",
            "ambient": "ambient",
            "dim": "ambient",
            "warm": "warm",
            "cozy": "warm"
        }
        if "light_preference" in fixed:
            val = fixed["light_preference"].lower().strip()
            fixed["light_preference"] = light_map.get(val, "natural")

        # Normalize clothing_insulation_typical
        clothing_map = {
            "very light": "very_light",
            "very_light": "very_light",
            "light": "light",
            "medium": "medium",
            "moderate": "medium",
            "warm": "warm",
            "heavy": "warm",
            "very warm": "very_warm",
            "very_warm": "very_warm"
        }
        if "clothing_insulation_typical" in fixed:
            val = fixed["clothing_insulation_typical"].lower().strip()
            fixed["clothing_insulation_typical"] = clothing_map.get(val, "medium")

        # Normalize workspace_variety
        variety_map = {
            "low": "low",
            "none": "low",
            "medium": "medium",
            "moderate": "medium",
            "high": "high"
        }
        if "workspace_variety" in fixed:
            val = fixed["workspace_variety"].lower().strip()
            fixed["workspace_variety"] = variety_map.get(val, "medium")

        return fixed

    def validate_and_fix(self, json_data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, List[str]]:
        """Validate and attempt to fix JSON.

        Returns:
            Tuple of (fixed_data, is_valid, remaining_errors)
        """
        # First try to fix common issues
        fixed = self.fix_common_issues(json_data)

        # Then validate
        is_valid, errors = self.validate(fixed)

        return fixed, is_valid, errors
