"""
Treatment Data Loader

Loads treatment recommendations from external data files.
Supports JSON format with fallback to hardcoded defaults.

Data Sources:
- Primary: data/treatments/disease_treatments.json
- Fallback: Hardcoded minimal recommendations

Recommended External Datasets:
1. PlantVillage (Penn State): https://plantvillage.psu.edu/
2. EPPO Global Database: https://gd.eppo.int/
3. UC Davis IPM: https://ipm.ucanr.edu/
4. CABI Crop Protection Compendium: https://www.cabi.org/cpc/
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TreatmentData:
    """Treatment data for a disease."""
    disease_name: str
    scientific_name: Optional[str] = None
    affected_crops: List[str] = field(default_factory=list)
    favorable_conditions: Optional[str] = None
    immediate_actions: List[str] = field(default_factory=list)
    organic_treatments: List[str] = field(default_factory=list)
    chemical_treatments: List[str] = field(default_factory=list)
    prevention_measures: List[str] = field(default_factory=list)
    monitoring_schedule: str = ""
    estimated_recovery: str = ""
    data_source: str = "Unknown"
    data_source_url: Optional[str] = None
    is_fallback: bool = False


@dataclass
class SeverityData:
    """Severity data for a disease."""
    base_severity: int
    spread_rate: str
    crop_impact: str
    is_fallback: bool = False


@dataclass
class RegionalData:
    """Regional restriction/notes data."""
    notes: str
    restricted_chemicals: List[str] = field(default_factory=list)
    preferred_approaches: List[str] = field(default_factory=list)


class TreatmentDataLoader:
    """
    Loads and manages treatment data from external files.

    Usage:
        loader = TreatmentDataLoader()
        treatment = loader.get_treatment("early blight")
        severity = loader.get_severity("early blight")
        regional = loader.get_regional_notes("US-CA")
    """

    # Default data file path
    DEFAULT_DATA_PATH = "data/treatments/disease_treatments.json"

    # Minimal fallback treatments (used when no data file exists)
    FALLBACK_TREATMENT = TreatmentData(
        disease_name="Unknown Disease",
        immediate_actions=[
            "⚠️ FALLBACK RESPONSE: No specific treatment data available for this disease",
            "Isolate affected plants to prevent spread",
            "Remove and destroy severely infected plant parts",
            "Photograph symptoms for expert consultation",
            "Contact local agricultural extension service for specific advice"
        ],
        organic_treatments=[
            "⚠️ FALLBACK: General organic recommendations",
            "Copper-based fungicide as broad-spectrum treatment",
            "Neem oil spray for fungal/pest issues",
            "Improve air circulation around plants"
        ],
        chemical_treatments=[
            "⚠️ FALLBACK: Consult local agricultural extension for specific recommendations",
            "Broad-spectrum fungicide may help if fungal origin suspected",
            "Always read and follow label directions"
        ],
        prevention_measures=[
            "⚠️ FALLBACK: Practice crop rotation",
            "Use disease-resistant varieties when replanting",
            "Maintain proper plant nutrition and watering",
            "Ensure good drainage"
        ],
        monitoring_schedule="⚠️ FALLBACK: Daily monitoring until disease is properly identified",
        estimated_recovery="⚠️ FALLBACK: Unknown - consult local experts for specific disease",
        data_source="⚠️ FALLBACK:  - No data file loaded",
        is_fallback=True
    )

    FALLBACK_SEVERITY = SeverityData(
        base_severity=50,
        spread_rate="unknown",
        crop_impact="Unknown - disease not in database",
        is_fallback=True
    )

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the treatment data loader.

        Args:
            data_path: Path to treatment data JSON file.
                      Defaults to data/treatments/disease_treatments.json
        """
        self.data_path = data_path or self.DEFAULT_DATA_PATH
        self._data: Dict[str, Any] = {}
        self._treatments: Dict[str, TreatmentData] = {}
        self._severities: Dict[str, SeverityData] = {}
        self._regional: Dict[str, RegionalData] = {}
        self._metadata: Dict[str, Any] = {}
        self._loaded = False
        self._load_error: Optional[str] = None

        # Try to load data
        self._load_data()

    def _load_data(self) -> bool:
        """Load treatment data from JSON file."""
        try:
            # Find the data file
            data_file = Path(self.data_path)

            # Try relative to current directory
            if not data_file.exists():
                # Try relative to app root
                app_root = Path(__file__).parent.parent.parent
                data_file = app_root / self.data_path

            if not data_file.exists():
                self._load_error = f"Data file not found: {self.data_path}"
                logger.warning(f"Treatment data file not found: {data_file}. Using fallback data.")
                return False

            # Load JSON
            with open(data_file, 'r', encoding='utf-8') as f:
                self._data = json.load(f)

            # Parse metadata
            self._metadata = self._data.get("_metadata", {})

            # Parse diseases
            diseases = self._data.get("diseases", {})
            for disease_name, disease_data in diseases.items():
                self._treatments[disease_name.lower()] = TreatmentData(
                    disease_name=disease_name,
                    scientific_name=disease_data.get("scientific_name"),
                    affected_crops=disease_data.get("affected_crops", []),
                    favorable_conditions=disease_data.get("favorable_conditions"),
                    immediate_actions=disease_data.get("immediate_actions", []),
                    organic_treatments=disease_data.get("organic_treatments", []),
                    chemical_treatments=disease_data.get("chemical_treatments", []),
                    prevention_measures=disease_data.get("prevention_measures", []),
                    monitoring_schedule=disease_data.get("monitoring_schedule", ""),
                    estimated_recovery=disease_data.get("estimated_recovery", ""),
                    data_source=disease_data.get("data_source", "Unknown"),
                    data_source_url=disease_data.get("data_source_url"),
                    is_fallback=False
                )

            # Parse severity data
            severity_data = self._data.get("severity_data", {})
            for disease_name, sev_data in severity_data.items():
                self._severities[disease_name.lower()] = SeverityData(
                    base_severity=sev_data.get("base_severity", 50),
                    spread_rate=sev_data.get("spread_rate", "unknown"),
                    crop_impact=sev_data.get("crop_impact", "Unknown"),
                    is_fallback=False
                )

            # Parse regional data
            regional_data = self._data.get("regional_restrictions", {})
            for region_code, reg_data in regional_data.items():
                self._regional[region_code] = RegionalData(
                    notes=reg_data.get("notes", ""),
                    restricted_chemicals=reg_data.get("restricted_chemicals", []),
                    preferred_approaches=reg_data.get("preferred_approaches", [])
                )

            self._loaded = True
            logger.info(f"Loaded treatment data: {len(self._treatments)} diseases, "
                       f"{len(self._severities)} severity entries, "
                       f"{len(self._regional)} regional notes")
            logger.info(f"Data sources: {self._metadata.get('sources', ['Unknown'])}")

            return True

        except json.JSONDecodeError as e:
            self._load_error = f"Invalid JSON in data file: {e}"
            logger.error(f"Failed to parse treatment data file: {e}")
            return False
        except Exception as e:
            self._load_error = f"Error loading data: {e}"
            logger.error(f"Failed to load treatment data: {e}")
            return False

    def get_treatment(self, disease_name: str) -> TreatmentData:
        """
        Get treatment data for a disease.

        Args:
            disease_name: Name of the disease (case-insensitive)

        Returns:
            TreatmentData with is_fallback=True if not found in database
        """
        # Normalize disease name
        normalized = disease_name.lower().strip()

        # Direct lookup
        if normalized in self._treatments:
            return self._treatments[normalized]

        # Try partial matching
        for key, treatment in self._treatments.items():
            if key in normalized or normalized in key:
                return treatment

        # Return fallback
        fallback = TreatmentData(
            disease_name=disease_name,
            immediate_actions=self.FALLBACK_TREATMENT.immediate_actions.copy(),
            organic_treatments=self.FALLBACK_TREATMENT.organic_treatments.copy(),
            chemical_treatments=self.FALLBACK_TREATMENT.chemical_treatments.copy(),
            prevention_measures=self.FALLBACK_TREATMENT.prevention_measures.copy(),
            monitoring_schedule=self.FALLBACK_TREATMENT.monitoring_schedule,
            estimated_recovery=self.FALLBACK_TREATMENT.estimated_recovery,
            data_source=f"FALLBACK - '{disease_name}' not found in database",
            is_fallback=True
        )
        return fallback

    def get_severity(self, disease_name: str) -> SeverityData:
        """
        Get severity data for a disease.

        Args:
            disease_name: Name of the disease (case-insensitive)

        Returns:
            SeverityData with is_fallback=True if not found
        """
        normalized = disease_name.lower().strip()

        # Direct lookup
        if normalized in self._severities:
            return self._severities[normalized]

        # Try partial matching
        for key, severity in self._severities.items():
            if key in normalized or normalized in key:
                return severity

        # Return fallback
        return SeverityData(
            base_severity=50,
            spread_rate="unknown",
            crop_impact=f"'{disease_name}' not in severity database - using default moderate severity",
            is_fallback=True
        )

    def get_regional_notes(self, region_code: str) -> Optional[RegionalData]:
        """
        Get regional notes for a region code.

        Args:
            region_code: Region code (e.g., "US-CA", "EU")

        Returns:
            RegionalData or None if not found
        """
        return self._regional.get(region_code)

    def get_all_diseases(self) -> List[str]:
        """Get list of all diseases in the database."""
        return list(self._treatments.keys())

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the loaded data."""
        return {
            "loaded": self._loaded,
            "load_error": self._load_error,
            "disease_count": len(self._treatments),
            "severity_count": len(self._severities),
            "regional_count": len(self._regional),
            "data_sources": self._metadata.get("sources", []),
            "last_updated": self._metadata.get("last_updated", "Unknown"),
            "version": self._metadata.get("version", "Unknown")
        }

    def is_loaded(self) -> bool:
        """Check if data was loaded successfully."""
        return self._loaded

    def reload(self, data_path: Optional[str] = None) -> bool:
        """
        Reload data from file.

        Args:
            data_path: Optional new data path

        Returns:
            True if loaded successfully
        """
        if data_path:
            self.data_path = data_path

        self._treatments.clear()
        self._severities.clear()
        self._regional.clear()
        self._loaded = False
        self._load_error = None

        return self._load_data()


# Singleton instance
_treatment_loader: Optional[TreatmentDataLoader] = None


def get_treatment_loader() -> TreatmentDataLoader:
    """Get the treatment data loader singleton."""
    global _treatment_loader
    if _treatment_loader is None:
        _treatment_loader = TreatmentDataLoader()
    return _treatment_loader
