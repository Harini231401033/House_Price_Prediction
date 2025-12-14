import joblib
import pandas as pd
import os
import argparse
import sys

model_path = os.path.join(os.path.dirname(__file__), "..", "house_price_model.pkl")

try:
    model = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {model_path}: {e}")

def predict(sample):
    df = pd.DataFrame([sample])
    return model.predict(df)[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--area", type=int)
    parser.add_argument("--bedrooms", type=int)
    parser.add_argument("--bathrooms", type=int)
    parser.add_argument("--year_built", type=int)
    parser.add_argument("--age", type=int)
    parser.add_argument("--location")
    parser.add_argument("--defaults", action="store_true", help="Use sample default values for missing inputs")
    args = parser.parse_args()

    def _get_int(prompt):
        try:
            val = input(f"{prompt}: ").strip()
        except EOFError:
            val = ""
        if not val:
            return None
        try:
            return int(val)
        except ValueError:
            print("Invalid input, using default.")
            return None

    defaults = {
        "area": 1648,
        "bedrooms": 4,
        "bathrooms": 1,
        "year_built": 1953,
        "age": 72,
        "location": "Delhi",
    }

    def _get_value(name, arg_val, cast=int):
        if arg_val is not None:
            return arg_val
        if args.defaults:
            return defaults[name]
        if not sys.stdin.isatty():
            raise SystemExit(f"Missing required argument --{name}; provide via flag or use --defaults")
        # interactive prompt
        if cast is int:
            v = _get_int(name)
            while v is None:
                print(f"Please enter a valid integer for {name} (try again)")
                v = _get_int(name)
            return v
        else:
            try:
                val = input(f"{name}: ").strip()
            except EOFError:
                val = ""
            while not val:
                print(f"Please enter a non-empty value for {name} (try again)")
                try:
                    val = input(f"{name}: ").strip()
                except EOFError:
                    val = ""
            return val

    area = _get_value("area", args.area, int)
    bedrooms = _get_value("bedrooms", args.bedrooms, int)
    bathrooms = _get_value("bathrooms", args.bathrooms, int)
    year_built = _get_value("year_built", args.year_built, int)
    age = _get_value("age", args.age, int)
    location = _get_value("location", args.location, str)

    sample = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "year_built": year_built,
        "age": age,
        "location": location
    }

    pred = predict(sample)
    try:
        pred = int(pred)
    except Exception:
        pass
    print("Predicted Price:", pred)
