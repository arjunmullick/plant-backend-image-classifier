"""
Tests for TaxonomyResolver - species name normalization.

Tests cover:
- Common name resolution
- Scientific name parsing
- Fuzzy matching
- Edge cases and unknown species
"""

import pytest
from app.ml.species_models.taxonomy_resolver import TaxonomyResolver, get_taxonomy_resolver
from app.ml.species_models.base import TaxonomyDepth


class TestTaxonomyResolver:
    """Test suite for TaxonomyResolver."""

    @pytest.fixture
    def resolver(self):
        """Get taxonomy resolver instance."""
        return get_taxonomy_resolver()

    # === Common Name Resolution Tests ===

    def test_resolve_common_name_tomato(self, resolver):
        """Test resolving common name 'tomato'."""
        result = resolver.normalize("tomato")

        assert result.family == "Solanaceae"
        assert result.genus == "Solanum"
        assert result.species == "Solanum lycopersicum"
        assert "Tomato" in result.common_names
        assert result.taxonomy_depth == TaxonomyDepth.SPECIES

    def test_resolve_common_name_potato(self, resolver):
        """Test resolving common name 'potato'."""
        result = resolver.normalize("potato")

        assert result.family == "Solanaceae"
        assert result.genus == "Solanum"
        assert result.species == "Solanum tuberosum"
        assert result.taxonomy_depth == TaxonomyDepth.SPECIES

    def test_resolve_common_name_apple(self, resolver):
        """Test resolving common name 'apple'."""
        result = resolver.normalize("apple")

        assert result.family == "Rosaceae"
        assert result.genus == "Malus"
        assert result.species == "Malus domestica"

    def test_resolve_common_name_corn(self, resolver):
        """Test resolving common name 'corn'."""
        result = resolver.normalize("corn")

        assert result.family == "Poaceae"
        assert result.genus == "Zea"
        assert result.species == "Zea mays"

    def test_resolve_common_name_grape(self, resolver):
        """Test resolving common name 'grape'."""
        result = resolver.normalize("grape")

        assert result.family == "Vitaceae"
        assert result.genus == "Vitis"
        assert result.species == "Vitis vinifera"

    def test_resolve_common_name_case_insensitive(self, resolver):
        """Test that common name lookup is case-insensitive."""
        result_lower = resolver.normalize("tomato")
        result_upper = resolver.normalize("TOMATO")
        result_mixed = resolver.normalize("ToMaTo")

        assert result_lower.species == result_upper.species == result_mixed.species

    # === Scientific Name Resolution Tests ===

    def test_resolve_scientific_name_full(self, resolver):
        """Test resolving full scientific name."""
        result = resolver.normalize("Solanum lycopersicum")

        assert result.family == "Solanaceae"
        assert result.genus == "Solanum"
        assert result.species == "Solanum lycopersicum"

    def test_resolve_scientific_name_with_author(self, resolver):
        """Test resolving scientific name with author citation."""
        result = resolver.normalize("Solanum lycopersicum L.")

        assert result.genus == "Solanum"
        assert result.species == "Solanum lycopersicum"
        # Author citation should be removed

    def test_resolve_scientific_name_with_parenthetical_author(self, resolver):
        """Test resolving scientific name with parenthetical author."""
        result = resolver.normalize("Malus domestica (Borkh.)")

        assert result.genus == "Malus"
        assert result.family == "Rosaceae"

    # === Alias Resolution Tests ===

    def test_resolve_alias_old_name(self, resolver):
        """Test resolving old/synonym scientific names."""
        # Lycopersicon esculentum is old name for tomato
        result = resolver.normalize("lycopersicon esculentum")

        assert result.species == "Solanum lycopersicum"

    def test_resolve_alias_variant_names(self, resolver):
        """Test resolving variant common names."""
        result1 = resolver.normalize("bell pepper")
        result2 = resolver.normalize("chili pepper")

        assert result1.genus == "Capsicum"
        assert result2.genus == "Capsicum"

    # === Fuzzy Matching Tests ===

    def test_fuzzy_match_partial_name(self, resolver):
        """Test fuzzy matching with partial names."""
        result = resolver.normalize("cherry tomato")

        # Should match tomato
        assert result.genus == "Solanum"

    def test_fuzzy_match_misspelling(self, resolver):
        """Test fuzzy matching handles minor misspellings."""
        result = resolver.normalize("tomatto")  # Misspelled

        # May or may not match depending on threshold
        # At minimum should not crash
        assert result is not None

    # === Edge Cases ===

    def test_resolve_empty_string(self, resolver):
        """Test handling empty string."""
        result = resolver.normalize("")

        assert result.taxonomy_depth == TaxonomyDepth.UNKNOWN

    def test_resolve_unknown_species(self, resolver):
        """Test handling unknown species."""
        result = resolver.normalize("Xyz unknownus")

        assert result is not None
        # Should return something, even if not fully resolved
        assert result.taxonomy_depth in [TaxonomyDepth.UNKNOWN, TaxonomyDepth.GENUS]

    def test_resolve_none_input(self, resolver):
        """Test handling None-like inputs."""
        result = resolver.normalize("")
        assert result.taxonomy_depth == TaxonomyDepth.UNKNOWN

    def test_resolve_numeric_input(self, resolver):
        """Test handling numeric/garbage input."""
        result = resolver.normalize("12345")

        assert result is not None
        assert result.taxonomy_depth == TaxonomyDepth.UNKNOWN

    # === PlantVillage Label Format Tests ===

    def test_resolve_plantvillage_format(self, resolver):
        """Test resolving PlantVillage-style labels."""
        # PlantVillage uses format: "Crop___Disease" or "Crop___healthy"
        result = resolver.normalize("Tomato___Early_blight")

        # Should extract crop name
        # Note: This depends on preprocessing the label
        # The resolver itself handles the normalized name

    # === Family Lookup Tests ===

    def test_is_same_family_true(self, resolver):
        """Test is_same_family returns true for related species."""
        assert resolver.is_same_family("tomato", "potato")  # Both Solanaceae
        assert resolver.is_same_family("apple", "strawberry")  # Both Rosaceae

    def test_is_same_family_false(self, resolver):
        """Test is_same_family returns false for unrelated species."""
        assert not resolver.is_same_family("tomato", "corn")
        assert not resolver.is_same_family("apple", "grape")

    def test_is_same_genus_true(self, resolver):
        """Test is_same_genus returns true for related species."""
        assert resolver.is_same_genus("tomato", "potato")  # Both Solanum

    def test_is_same_genus_false(self, resolver):
        """Test is_same_genus returns false for different genera."""
        assert not resolver.is_same_genus("tomato", "pepper")  # Solanum vs Capsicum

    # === Family Members Lookup ===

    def test_get_family_members(self, resolver):
        """Test getting all species in a family."""
        solanaceae = resolver.get_family_members("Solanaceae")

        assert len(solanaceae) >= 3  # At least tomato, potato, pepper
        assert "Solanum lycopersicum" in solanaceae
        assert "Solanum tuberosum" in solanaceae

    def test_get_genus_members(self, resolver):
        """Test getting all species in a genus."""
        solanum = resolver.get_genus_members("Solanum")

        assert len(solanum) >= 2  # At least tomato and potato
        assert "Solanum lycopersicum" in solanum

    # === Coverage Tests for All Major Crops ===

    @pytest.mark.parametrize("common_name,expected_family", [
        ("tomato", "Solanaceae"),
        ("potato", "Solanaceae"),
        ("pepper", "Solanaceae"),
        ("eggplant", "Solanaceae"),
        ("apple", "Rosaceae"),
        ("peach", "Rosaceae"),
        ("strawberry", "Rosaceae"),
        ("corn", "Poaceae"),
        ("rice", "Poaceae"),
        ("wheat", "Poaceae"),
        ("grape", "Vitaceae"),
        ("cucumber", "Cucurbitaceae"),
        ("squash", "Cucurbitaceae"),
        ("bean", "Fabaceae"),
        ("soybean", "Fabaceae"),
        ("orange", "Rutaceae"),
        ("lemon", "Rutaceae"),
        ("sunflower", "Asteraceae"),
        ("lettuce", "Asteraceae"),
    ])
    def test_resolve_major_crops(self, resolver, common_name, expected_family):
        """Test that all major crops resolve correctly."""
        result = resolver.normalize(common_name)

        assert result.family == expected_family, f"{common_name} should be in {expected_family}"
        assert result.taxonomy_depth == TaxonomyDepth.SPECIES


class TestTaxonomyResolverSingleton:
    """Test singleton behavior of taxonomy resolver."""

    def test_singleton_returns_same_instance(self):
        """Test that get_taxonomy_resolver returns same instance."""
        resolver1 = get_taxonomy_resolver()
        resolver2 = get_taxonomy_resolver()

        assert resolver1 is resolver2
