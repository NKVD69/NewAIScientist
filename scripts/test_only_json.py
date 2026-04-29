import asyncio
from co_scientist import _parse_json_response
from test_suite import TestUtilities

def main():
    print("Testing TestUtilities.test_json_parsing()...")
    try:
        TestUtilities.test_json_parsing()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
