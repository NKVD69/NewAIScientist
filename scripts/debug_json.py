
from co_scientist import _parse_json_response


def test():
    print("Testing _parse_json_response...")
    junk_json = "Here is the result: {\"success\": true} Hope this helps!"
    try:
        parsed = _parse_json_response(junk_json)
        print(f"Parsed: {parsed}")
        assert parsed == {"success": True}
        print("✓ Success")
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    test()
