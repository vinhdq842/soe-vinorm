from soe_vinorm.nsw_expander import RuleBasedNSWExpander


class TestRuleBasedNSWExpander:
    """Test cases for RuleBasedNSWExpander class."""

    def test_init_with_custom_dicts(self, vn_dict, abbr_dict):
        """Test expander initialization with custom dictionaries."""
        expander = RuleBasedNSWExpander(vn_dict=vn_dict, abbr_dict=abbr_dict)
        assert expander._vn_dict == set(vn_dict)
        assert expander._abbr_dict == abbr_dict

    def test_init_with_default_dicts(self):
        """Test expander initialization with default dictionaries."""
        expander = RuleBasedNSWExpander()
        assert len(expander._vn_dict) > 0
        assert len(expander._abbr_dict) > 0

    def test_expand_with_urle_arg(self):
        """Test expander with urle."""
        expander = RuleBasedNSWExpander(expand_url=True)
        result = expander.expand(["https://www.example.com"], ["B-URLE"])
        assert result != ["https://www.example.com"]

        expander = RuleBasedNSWExpander(expand_url=False)
        result = expander.expand(["https://www.example.com"], ["B-URLE"])
        assert result == ["https://www.example.com"]
