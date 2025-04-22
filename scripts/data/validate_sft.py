import json


def validate_json_structure(json_data):
    """
    Validate the structure of the given JSON data, ensuring all required keys are present.
    """
    required_top_level_keys = {"system", "instruction", "input", "output"}
    errors = []

    for i, entry in enumerate(json_data):
        # Check top-level keys
        if not all(key in entry for key in required_top_level_keys):
            missing_keys = required_top_level_keys - entry.keys()
            errors.append(f"Entry {i} is missing keys: {missing_keys}")

    return errors

# Load JSON file


def main():
    try:
        with open("F:\CS\AI\DistilledRM\data/arena_et_cot_distill.json", "r", encoding="utf-8") as file:
            json_data = json.load(file)

        errors = validate_json_structure(json_data)

        if errors:
            print("The JSON file has the following issues:")
            for error in errors:
                print(f"- {error}")
        else:
            print("The JSON file is valid and conforms to the specified format.")

    except json.JSONDecodeError as e:
        print(f"Invalid JSON file: {e}")
    except FileNotFoundError:
        print("The file is not found. Make sure it exists in the current directory.")


if __name__ == "__main__":
    main()

