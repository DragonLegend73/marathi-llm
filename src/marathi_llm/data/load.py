"""Data loading and standardized extraction across Marathi instruction datasets."""

import os
import sys
from typing import Generator, Dict, Any, Optional, List
from datasets import load_dataset


def extract_s0_alpaca_row(row: Dict[str, Any], row_idx: int) -> Optional[Dict[str, Any]]:
    """Extracts cleaned Marathi instruction and response from S0 Alpaca row."""
    # S0 has 'translated_text' formatted as:
    # "खाली एक सूचना आहे ... ### सूचना:\n...\n\n### प्रतिसाद:\n..."
    trans = row.get("translated_text", "")
    instruction = ""
    input_text = ""
    response = ""

    if trans and "### प्रतिसाद:" in trans:
        parts = trans.split("### प्रतिसाद:")
        response = parts[1].strip() if len(parts) > 1 else ""
        inst_part = parts[0]
        if "### सूचना:" in inst_part:
            raw_inst = inst_part.split("### सूचना:")[1].strip()
            if "### इनपुट:" in raw_inst:
                in_parts = raw_inst.split("### इनपुट:")
                instruction = in_parts[0].strip()
                input_text = in_parts[1].strip()
            else:
                instruction = raw_inst
    else:
        # If translated_text lacks proper Marathi markers, do not fall back to English raw fields
        return None

    if not instruction or not response:
        return None

    PLACEHOLDER_INPUTS = {
        "इनपुट नाही",
        "इनपुट नाही.",
        "noinput",
        "no input",
        "none",
        "नाही",
        "n/a",
    }
    if input_text.lower().strip() in PLACEHOLDER_INPUTS:
        input_text = ""

    return {
        "id": f"s0_{row_idx}",
        "row_id": row_idx,
        "instruction": instruction,
        "input": input_text,
        "response": response,
        "source": "S0_alpaca",
    }


def stream_s0_alpaca(dataset_name: str = "rachittshah/alpaca-marahti") -> Generator[Dict[str, Any], None, None]:
    """Streams rows from S0 Alpaca dataset."""
    ds = load_dataset(dataset_name, split="train")
    for idx, row in enumerate(ds):
        extracted = extract_s0_alpaca_row(row, idx)
        if extracted:
            yield extracted


def stream_s1_indicalign(
    dataset_name: str = "ai4bharat/indic-align",
    configs: Optional[List[str]] = None,
    streaming: bool = True,
) -> Generator[Dict[str, Any], None, None]:
    """Streams Marathi rows from IndicAlign-Instruct across configs."""
    target_configs = configs or [
        "Dolly_T",
        "OpenAssistant_T",
        "WikiHow",
        "Anudesh",
        "Wiki_Conv",
        "Wiki_Chat",
        "HHRLHF_T",
    ]

    for cfg in target_configs:
        try:
            ds = load_dataset(dataset_name, cfg, split="train", streaming=streaming)
            for idx, row in enumerate(ds):
                mar_turns = row.get("mar_Deva", [])
                if not mar_turns or not isinstance(mar_turns, list):
                    continue

                # Each turn in mar_Deva is [user_turn, assistant_turn]
                for turn_idx, turn in enumerate(mar_turns):
                    if len(turn) >= 2:
                        u_msg = str(turn[0]).strip()
                        a_msg = str(turn[1]).strip()
                        if u_msg and a_msg:
                            yield {
                                "id": f"s1_{cfg}_{idx}_{turn_idx}",
                                "instruction": u_msg,
                                "input": "",
                                "response": a_msg,
                                "source": f"S1_{cfg}",
                            }
        except Exception as e:
            print(f"Warning: could not stream IndicAlign config '{cfg}': {e}")


def stream_s2_sandlogic(
    dataset_name: str = "SandLogicTechnologies/Indic_Chat_Dataset",
    split: str = "marathi",
    limit: Optional[int] = 200000,
    streaming: bool = True,
) -> Generator[Dict[str, Any], None, None]:
    """Streams Marathi rows from SandLogic Indic Chat Dataset."""
    try:
        ds = load_dataset(dataset_name, split=split, streaming=streaming)
        count = 0
        for idx, row in enumerate(ds):
            msgs = row.get("messages", [])
            if not msgs or not isinstance(msgs, list):
                continue

            u_msg = ""
            a_msg = ""
            for m in msgs:
                role = m.get("role", "")
                content = m.get("content", "").strip()
                if role == "user" and not u_msg:
                    u_msg = content
                elif role == "assistant" and u_msg and not a_msg:
                    a_msg = content
                    break

            if u_msg and a_msg:
                yield {
                    "id": f"s2_{idx}",
                    "instruction": u_msg,
                    "input": "",
                    "response": a_msg,
                    "source": "S2_sandlogic",
                }
                count += 1
                if limit and count >= limit:
                    break
    except Exception as e:
        print(f"Warning: could not stream SandLogic dataset: {e}")


def stream_s3_aya(
    dataset_name: str = "CohereForAI/aya_dataset",
    split: str = "train",
    streaming: bool = True,
) -> Generator[Dict[str, Any], None, None]:
    """Streams Marathi rows from Cohere Aya Dataset."""
    try:
        ds = load_dataset(dataset_name, split=split, streaming=streaming)
        idx = 0
        for row in ds:
            lang = row.get("language_code", "") or row.get("language", "")
            if lang in ["mar", "mr", "Marathi"]:
                u_msg = str(row.get("inputs", "")).strip()
                a_msg = str(row.get("targets", "")).strip()
                if u_msg and a_msg:
                    yield {
                        "id": f"s3_{idx}",
                        "instruction": u_msg,
                        "input": "",
                        "response": a_msg,
                        "source": "S3_aya",
                    }
                    idx += 1
    except Exception as e:
        print(f"Warning: could not stream Aya dataset: {e}")
