Prompt Structure Variants for Review
=============================================

Base reference (already in use): detail_notemplate
Added variants in this package:

- tagged_sections (hf_chat_messages_tagged_sections_v1): Tag-delimited sections ([TASK]/[CONTEXT]/[CONSTRAINTS]) to create clear boundaries and reduce prompt ambiguity.
- json_payload (hf_chat_messages_json_payload_v1): JSON-style context payload emphasizing machine-readable slot-value alignment and explicit decision policy.

Files include train/val/test JSONL + matching review CSV for each variant.
Assistant targets are unchanged; only system/user prompt structure is modified.
