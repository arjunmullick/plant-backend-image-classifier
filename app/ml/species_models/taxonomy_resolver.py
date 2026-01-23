"""
Taxonomy Resolver

Normalizes species names from different models into a consistent format.
Uses local cache for common crops and can query external APIs for unknown species.

Data sources:
- Local taxonomy cache (common agricultural plants)
- GBIF Backbone Taxonomy API (fallback for unknown species)
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from app.ml.species_models.base import NormalizedTaxonomy, TaxonomyDepth

logger = logging.getLogger(__name__)


@dataclass
class TaxonomyEntry:
    """Single entry in taxonomy database."""
    family: str
    genus: str
    species: str  # Binomial name
    common_names: List[str]
    gbif_id: Optional[int] = None
    aliases: List[str] = None  # Alternative names/spellings

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


class TaxonomyResolver:
    """
    Resolves and normalizes taxonomy names from various model outputs.

    Features:
    - Common name to scientific name mapping
    - Scientific name normalization (removes author citations)
    - Fuzzy matching for misspellings
    - Local cache for fast lookup
    - GBIF API fallback for unknown species
    """

    # Local taxonomy database for common agricultural plants
    TAXONOMY_DATABASE: Dict[str, TaxonomyEntry] = {
        # Solanaceae (Nightshade family)
        "solanum lycopersicum": TaxonomyEntry(
            family="Solanaceae", genus="Solanum", species="Solanum lycopersicum",
            common_names=["Tomato", "Cherry Tomato", "Roma Tomato", "Beefsteak Tomato"],
            aliases=["tomato", "lycopersicon esculentum", "solanum lycopersicum l."]
        ),
        "solanum tuberosum": TaxonomyEntry(
            family="Solanaceae", genus="Solanum", species="Solanum tuberosum",
            common_names=["Potato", "Irish Potato", "White Potato"],
            aliases=["potato", "solanum tuberosum l."]
        ),
        "capsicum annuum": TaxonomyEntry(
            family="Solanaceae", genus="Capsicum", species="Capsicum annuum",
            common_names=["Bell Pepper", "Chili Pepper", "Sweet Pepper", "Jalapeño"],
            aliases=["pepper", "bell pepper", "capsicum annuum l."]
        ),
        "solanum melongena": TaxonomyEntry(
            family="Solanaceae", genus="Solanum", species="Solanum melongena",
            common_names=["Eggplant", "Aubergine", "Brinjal"],
            aliases=["eggplant", "aubergine", "solanum melongena l."]
        ),

        # Rosaceae (Rose family)
        "malus domestica": TaxonomyEntry(
            family="Rosaceae", genus="Malus", species="Malus domestica",
            common_names=["Apple", "Common Apple", "Cultivated Apple"],
            aliases=["apple", "malus pumila", "malus domestica borkh."]
        ),
        "prunus persica": TaxonomyEntry(
            family="Rosaceae", genus="Prunus", species="Prunus persica",
            common_names=["Peach", "Nectarine"],
            aliases=["peach", "prunus persica (l.) batsch"]
        ),
        "fragaria × ananassa": TaxonomyEntry(
            family="Rosaceae", genus="Fragaria", species="Fragaria × ananassa",
            common_names=["Strawberry", "Garden Strawberry"],
            aliases=["strawberry", "fragaria ananassa"]
        ),
        "rubus idaeus": TaxonomyEntry(
            family="Rosaceae", genus="Rubus", species="Rubus idaeus",
            common_names=["Raspberry", "Red Raspberry"],
            aliases=["raspberry", "rubus idaeus l."]
        ),
        "pyrus communis": TaxonomyEntry(
            family="Rosaceae", genus="Pyrus", species="Pyrus communis",
            common_names=["Pear", "Common Pear", "European Pear"],
            aliases=["pear", "pyrus communis l."]
        ),
        "prunus avium": TaxonomyEntry(
            family="Rosaceae", genus="Prunus", species="Prunus avium",
            common_names=["Cherry", "Sweet Cherry", "Wild Cherry"],
            aliases=["cherry", "sweet cherry", "prunus avium l."]
        ),

        # Poaceae (Grass family)
        "zea mays": TaxonomyEntry(
            family="Poaceae", genus="Zea", species="Zea mays",
            common_names=["Corn", "Maize", "Sweet Corn"],
            aliases=["corn", "maize", "zea mays l."]
        ),
        "oryza sativa": TaxonomyEntry(
            family="Poaceae", genus="Oryza", species="Oryza sativa",
            common_names=["Rice", "Asian Rice", "Paddy Rice"],
            aliases=["rice", "oryza sativa l."]
        ),
        "triticum aestivum": TaxonomyEntry(
            family="Poaceae", genus="Triticum", species="Triticum aestivum",
            common_names=["Wheat", "Common Wheat", "Bread Wheat"],
            aliases=["wheat", "triticum aestivum l."]
        ),

        # Vitaceae (Grape family)
        "vitis vinifera": TaxonomyEntry(
            family="Vitaceae", genus="Vitis", species="Vitis vinifera",
            common_names=["Grape", "Common Grape", "Wine Grape"],
            aliases=["grape", "vitis vinifera l."]
        ),

        # Cucurbitaceae (Gourd family)
        "cucumis sativus": TaxonomyEntry(
            family="Cucurbitaceae", genus="Cucumis", species="Cucumis sativus",
            common_names=["Cucumber"],
            aliases=["cucumber", "cucumis sativus l."]
        ),
        "cucurbita pepo": TaxonomyEntry(
            family="Cucurbitaceae", genus="Cucurbita", species="Cucurbita pepo",
            common_names=["Squash", "Zucchini", "Pumpkin", "Summer Squash"],
            aliases=["squash", "zucchini", "pumpkin", "cucurbita pepo l."]
        ),
        "citrullus lanatus": TaxonomyEntry(
            family="Cucurbitaceae", genus="Citrullus", species="Citrullus lanatus",
            common_names=["Watermelon"],
            aliases=["watermelon", "citrullus lanatus (thunb.) matsum. & nakai"]
        ),

        # Fabaceae (Legume family)
        "phaseolus vulgaris": TaxonomyEntry(
            family="Fabaceae", genus="Phaseolus", species="Phaseolus vulgaris",
            common_names=["Common Bean", "Kidney Bean", "Green Bean", "String Bean"],
            aliases=["bean", "common bean", "phaseolus vulgaris l."]
        ),
        "glycine max": TaxonomyEntry(
            family="Fabaceae", genus="Glycine", species="Glycine max",
            common_names=["Soybean", "Soya Bean"],
            aliases=["soybean", "soya", "glycine max (l.) merr."]
        ),
        "pisum sativum": TaxonomyEntry(
            family="Fabaceae", genus="Pisum", species="Pisum sativum",
            common_names=["Pea", "Garden Pea", "Green Pea"],
            aliases=["pea", "pisum sativum l."]
        ),

        # Rutaceae (Citrus family)
        "citrus × sinensis": TaxonomyEntry(
            family="Rutaceae", genus="Citrus", species="Citrus × sinensis",
            common_names=["Orange", "Sweet Orange", "Navel Orange"],
            aliases=["orange", "citrus sinensis"]
        ),
        "citrus limon": TaxonomyEntry(
            family="Rutaceae", genus="Citrus", species="Citrus limon",
            common_names=["Lemon"],
            aliases=["lemon", "citrus limon (l.) osbeck"]
        ),
        "citrus × paradisi": TaxonomyEntry(
            family="Rutaceae", genus="Citrus", species="Citrus × paradisi",
            common_names=["Grapefruit"],
            aliases=["grapefruit", "citrus paradisi"]
        ),

        # Asteraceae (Daisy family)
        "helianthus annuus": TaxonomyEntry(
            family="Asteraceae", genus="Helianthus", species="Helianthus annuus",
            common_names=["Sunflower", "Common Sunflower"],
            aliases=["sunflower", "helianthus annuus l."]
        ),
        "lactuca sativa": TaxonomyEntry(
            family="Asteraceae", genus="Lactuca", species="Lactuca sativa",
            common_names=["Lettuce", "Garden Lettuce"],
            aliases=["lettuce", "lactuca sativa l."]
        ),

        # Brassicaceae (Mustard/Cabbage family)
        "brassica oleracea": TaxonomyEntry(
            family="Brassicaceae", genus="Brassica", species="Brassica oleracea",
            common_names=["Cabbage", "Broccoli", "Cauliflower", "Kale", "Brussels Sprouts"],
            aliases=["cabbage", "broccoli", "cauliflower", "brassica oleracea l."]
        ),
        "brassica rapa": TaxonomyEntry(
            family="Brassicaceae", genus="Brassica", species="Brassica rapa",
            common_names=["Turnip", "Chinese Cabbage", "Bok Choy"],
            aliases=["turnip", "brassica rapa l."]
        ),

        # Amaranthaceae
        "spinacia oleracea": TaxonomyEntry(
            family="Amaranthaceae", genus="Spinacia", species="Spinacia oleracea",
            common_names=["Spinach"],
            aliases=["spinach", "spinacia oleracea l."]
        ),

        # Apiaceae (Carrot family)
        "daucus carota": TaxonomyEntry(
            family="Apiaceae", genus="Daucus", species="Daucus carota",
            common_names=["Carrot", "Wild Carrot"],
            aliases=["carrot", "daucus carota l."]
        ),

        # Alliaceae/Amaryllidaceae
        "allium cepa": TaxonomyEntry(
            family="Amaryllidaceae", genus="Allium", species="Allium cepa",
            common_names=["Onion", "Common Onion"],
            aliases=["onion", "allium cepa l."]
        ),
        "allium sativum": TaxonomyEntry(
            family="Amaryllidaceae", genus="Allium", species="Allium sativum",
            common_names=["Garlic"],
            aliases=["garlic", "allium sativum l."]
        ),

        # Musaceae
        "musa acuminata": TaxonomyEntry(
            family="Musaceae", genus="Musa", species="Musa acuminata",
            common_names=["Banana", "Cavendish Banana"],
            aliases=["banana", "musa acuminata colla"]
        ),

        # Convolvulaceae
        "ipomoea batatas": TaxonomyEntry(
            family="Convolvulaceae", genus="Ipomoea", species="Ipomoea batatas",
            common_names=["Sweet Potato", "Yam"],
            aliases=["sweet potato", "ipomoea batatas (l.) lam."]
        ),
    }

    # Common name to species mapping (lowercase)
    COMMON_NAME_INDEX: Dict[str, str] = {}

    # Alias to species mapping (lowercase)
    ALIAS_INDEX: Dict[str, str] = {}

    def __init__(self, cache_path: Optional[str] = None):
        """
        Initialize taxonomy resolver.

        Args:
            cache_path: Path to additional taxonomy cache JSON file
        """
        self._build_indices()

        if cache_path:
            self._load_cache(cache_path)

    def _build_indices(self):
        """Build search indices from taxonomy database."""
        for species_key, entry in self.TAXONOMY_DATABASE.items():
            # Index common names
            for common_name in entry.common_names:
                self.COMMON_NAME_INDEX[common_name.lower()] = species_key

            # Index aliases
            for alias in entry.aliases:
                self.ALIAS_INDEX[alias.lower()] = species_key

    def _load_cache(self, cache_path: str):
        """Load additional taxonomy data from JSON cache."""
        try:
            path = Path(cache_path)
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                    for key, entry_data in data.items():
                        entry = TaxonomyEntry(**entry_data)
                        self.TAXONOMY_DATABASE[key.lower()] = entry
                self._build_indices()
                logger.info(f"Loaded {len(data)} additional taxonomy entries from cache")
        except Exception as e:
            logger.warning(f"Failed to load taxonomy cache: {e}")

    def normalize(
        self,
        name: str,
        source_model: Optional[str] = None
    ) -> NormalizedTaxonomy:
        """
        Normalize a species name to standard taxonomy.

        Args:
            name: Species name (common name, scientific name, or raw label)
            source_model: Model that produced this name (for logging)

        Returns:
            NormalizedTaxonomy with standardized fields
        """
        if not name:
            return NormalizedTaxonomy(taxonomy_depth=TaxonomyDepth.UNKNOWN)

        original_name = name
        name_lower = self._normalize_string(name)

        # Try exact match in database
        if name_lower in self.TAXONOMY_DATABASE:
            return self._entry_to_taxonomy(
                self.TAXONOMY_DATABASE[name_lower],
                original_name
            )

        # Try common name lookup
        if name_lower in self.COMMON_NAME_INDEX:
            species_key = self.COMMON_NAME_INDEX[name_lower]
            return self._entry_to_taxonomy(
                self.TAXONOMY_DATABASE[species_key],
                original_name
            )

        # Try alias lookup
        if name_lower in self.ALIAS_INDEX:
            species_key = self.ALIAS_INDEX[name_lower]
            return self._entry_to_taxonomy(
                self.TAXONOMY_DATABASE[species_key],
                original_name
            )

        # Try fuzzy matching
        fuzzy_match = self._fuzzy_match(name_lower)
        if fuzzy_match:
            return self._entry_to_taxonomy(
                self.TAXONOMY_DATABASE[fuzzy_match],
                original_name
            )

        # Try to parse as scientific name
        parsed = self._parse_scientific_name(name)
        if parsed:
            return parsed

        # Unknown - return with what we can extract
        logger.debug(f"Could not resolve taxonomy for: {name} (from {source_model})")
        return NormalizedTaxonomy(
            species=name,
            common_names=[name],
            taxonomy_depth=TaxonomyDepth.UNKNOWN,
            source_label=original_name
        )

    def _normalize_string(self, s: str) -> str:
        """Normalize string for matching."""
        # Remove author citations (e.g., "L.", "(Mill.)", "Borkh.")
        s = re.sub(r'\s*\([^)]*\)\s*', ' ', s)
        s = re.sub(r'\s+[A-Z][a-z]*\.?\s*$', '', s)

        # Normalize whitespace and case
        s = ' '.join(s.lower().split())

        # Remove special characters
        s = re.sub(r'[×]', '', s)

        return s.strip()

    def _fuzzy_match(self, name: str, threshold: float = 0.8) -> Optional[str]:
        """
        Find best fuzzy match in database.

        Uses simple substring and word matching.
        For production, consider using rapidfuzz library.
        """
        name_words = set(name.split())

        best_match = None
        best_score = 0.0

        # Check against all entries
        for species_key, entry in self.TAXONOMY_DATABASE.items():
            # Check against species key
            key_words = set(species_key.split())
            word_overlap = len(name_words & key_words) / max(len(name_words), len(key_words))

            if word_overlap > best_score and word_overlap >= threshold:
                best_score = word_overlap
                best_match = species_key

            # Check against common names
            for common_name in entry.common_names:
                common_lower = common_name.lower()
                if name in common_lower or common_lower in name:
                    if len(common_lower) / max(len(name), len(common_lower)) > best_score:
                        best_score = len(common_lower) / max(len(name), len(common_lower))
                        best_match = species_key

        return best_match

    def _parse_scientific_name(self, name: str) -> Optional[NormalizedTaxonomy]:
        """
        Parse a scientific name that's not in our database.

        Attempts to extract genus and species from binomial nomenclature.
        """
        # Clean up the name
        clean = self._normalize_string(name)
        parts = clean.split()

        if len(parts) >= 2:
            # Assume first word is genus, second is species epithet
            genus = parts[0].capitalize()
            species_epithet = parts[1].lower()
            binomial = f"{genus} {species_epithet}"

            # Try to find family from genus
            family = self._lookup_family_by_genus(genus)

            return NormalizedTaxonomy(
                family=family,
                genus=genus,
                species=binomial,
                taxonomy_depth=TaxonomyDepth.SPECIES if family else TaxonomyDepth.GENUS,
                source_label=name
            )

        elif len(parts) == 1:
            # Single word - could be genus or common name
            word = parts[0].capitalize()
            family = self._lookup_family_by_genus(word)

            if family:
                return NormalizedTaxonomy(
                    family=family,
                    genus=word,
                    taxonomy_depth=TaxonomyDepth.GENUS,
                    source_label=name
                )

        return None

    def _lookup_family_by_genus(self, genus: str) -> Optional[str]:
        """Look up family from genus name."""
        genus_lower = genus.lower()
        for entry in self.TAXONOMY_DATABASE.values():
            if entry.genus.lower() == genus_lower:
                return entry.family
        return None

    def _entry_to_taxonomy(
        self,
        entry: TaxonomyEntry,
        original_name: str
    ) -> NormalizedTaxonomy:
        """Convert database entry to NormalizedTaxonomy."""
        return NormalizedTaxonomy(
            family=entry.family,
            genus=entry.genus,
            species=entry.species,
            common_names=entry.common_names.copy(),
            taxonomy_depth=TaxonomyDepth.SPECIES,
            source_label=original_name,
            gbif_id=entry.gbif_id
        )

    def get_family_members(self, family: str) -> List[str]:
        """Get all species in a family."""
        family_lower = family.lower()
        return [
            entry.species
            for entry in self.TAXONOMY_DATABASE.values()
            if entry.family.lower() == family_lower
        ]

    def get_genus_members(self, genus: str) -> List[str]:
        """Get all species in a genus."""
        genus_lower = genus.lower()
        return [
            entry.species
            for entry in self.TAXONOMY_DATABASE.values()
            if entry.genus.lower() == genus_lower
        ]

    def is_same_family(self, name1: str, name2: str) -> bool:
        """Check if two species are in the same family."""
        tax1 = self.normalize(name1)
        tax2 = self.normalize(name2)
        return (
            tax1.family is not None and
            tax2.family is not None and
            tax1.family.lower() == tax2.family.lower()
        )

    def is_same_genus(self, name1: str, name2: str) -> bool:
        """Check if two species are in the same genus."""
        tax1 = self.normalize(name1)
        tax2 = self.normalize(name2)
        return (
            tax1.genus is not None and
            tax2.genus is not None and
            tax1.genus.lower() == tax2.genus.lower()
        )


# Singleton instance
_taxonomy_resolver: Optional[TaxonomyResolver] = None


def get_taxonomy_resolver() -> TaxonomyResolver:
    """Get singleton taxonomy resolver instance."""
    global _taxonomy_resolver
    if _taxonomy_resolver is None:
        # Try to load cache from default location
        cache_path = Path(__file__).parent / "data" / "taxonomy_cache.json"
        _taxonomy_resolver = TaxonomyResolver(
            cache_path=str(cache_path) if cache_path.exists() else None
        )
    return _taxonomy_resolver
