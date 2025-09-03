from state.quark_state_system.prompt_guardian import PromptGuardian


def test_rule_edit_blocked():
    pg = PromptGuardian()
    ctx = {
        "action_type": "edit",
        "target_files": [".cursor/rules/architecture-and-quality.mdc"],
        # user_confirmed omitted
    }
    allowed = pg.validate_action("edit_file", ctx)
    assert not allowed, "PromptGuardian should block rule-file edits without user confirmation"
