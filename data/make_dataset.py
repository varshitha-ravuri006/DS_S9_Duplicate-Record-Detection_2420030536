"""
make_dataset.py
---------------
Creates the sample customer dataset used by this project.

We cannot use a real customer database (privacy + we would not know the
correct answers), so we generate one ourselves. The big advantage of
generating the data is that we KNOW which records belong to the same person,
so we can write an honest ground_truth.csv and actually measure precision
and recall later.

The generator builds three kinds of records:

1. Unique records          -> one person, one row
2. Duplicate records       -> same person, written differently
                              (case, spacing, abbreviations, typos, +91, ...)
3. "Confusable" records    -> DIFFERENT people who look similar
                              (same name in the same city, siblings sharing
                              an address, etc.)

Group 3 matters a lot. Without it every similar-looking pair would be a true
duplicate and our evaluation scores would be meaninglessly high.

Run:  python data/make_dataset.py
"""

import random
from pathlib import Path

import pandas as pd

SEED = 42
N_ENTITIES = 330          # number of distinct real people
N_DUPLICATED = 75         # how many of them appear more than once
N_CONFUSABLE = 25         # pairs of different people who look alike

DATA_DIR = Path(__file__).resolve().parent

rng = random.Random(SEED)


# ----------------------------------------------------------------------
# 1. Raw ingredients
# ----------------------------------------------------------------------

FIRST_NAMES = [
    "Ravi", "Anil", "Sunil", "Priya", "Deepak", "Kavya", "Arjun", "Meera",
    "Rahul", "Sneha", "Vikram", "Pooja", "Karthik", "Divya", "Manoj", "Swathi",
    "Ajay", "Nisha", "Suresh", "Lakshmi", "Rohit", "Anjali", "Naveen", "Shreya",
    "Kiran", "Rekha", "Sandeep", "Geetha", "Praveen", "Bhavya", "Harsha",
    "Madhuri", "Venkat", "Ananya", "Gopal", "Indira", "Nikhil", "Sravani",
    "Prakash", "Aarti", "Chandra", "Neha", "Mahesh", "Ritu", "Sagar", "Vani",
    "Tarun", "Pallavi", "Yogesh", "Shilpa", "Balaji", "Keerthi", "Ramesh",
    "Sowmya", "Dinesh", "Asha", "Girish", "Namrata", "Abhishek", "Varsha",
]

LAST_NAMES = [
    "Kumar", "Sharma", "Reddy", "Rao", "Patel", "Nair", "Menon", "Iyer",
    "Gupta", "Verma", "Singh", "Joshi", "Naidu", "Chowdary", "Pillai",
    "Desai", "Bhat", "Kulkarni", "Malhotra", "Agarwal", "Mishra", "Bose",
    "Sinha", "Prasad", "Shetty", "Rathore", "Dutta", "Kapoor", "Saxena", "Jain",
]

# (canonical city, [ways people write it], [plausible pincodes], [local areas])
CITIES = [
    ("Hyderabad", ["Hyderabad", "HYD", "Hyd", "hyderabad", "Hyderabad "],
     ["500001", "500018", "500032", "500081"],
     ["Madhapur", "Kondapur", "Gachibowli", "Banjara Hills", "Kukatpally"]),
    ("Bengaluru", ["Bengaluru", "Bangalore", "BLR", "bengaluru", "Banglore"],
     ["560001", "560034", "560066", "560103"],
     ["Indiranagar", "Koramangala", "Whitefield", "Jayanagar", "HSR Layout"]),
    ("Mumbai", ["Mumbai", "MUM", "Bombay", "mumbai"],
     ["400001", "400053", "400072", "400703"],
     ["Andheri", "Bandra", "Powai", "Dadar", "Vashi"]),
    ("Chennai", ["Chennai", "MAA", "Madras", "chennai"],
     ["600001", "600028", "600096", "600119"],
     ["Adyar", "Velachery", "T Nagar", "Anna Nagar", "Guindy"]),
    ("Pune", ["Pune", "PNQ", "pune", "Poona"],
     ["411001", "411014", "411045", "411057"],
     ["Kothrud", "Hinjewadi", "Viman Nagar", "Baner", "Kharadi"]),
    ("Delhi", ["Delhi", "New Delhi", "DEL", "delhi"],
     ["110001", "110019", "110065", "110092"],
     ["Saket", "Dwarka", "Rohini", "Karol Bagh", "Vasant Kunj"]),
    ("Kolkata", ["Kolkata", "CCU", "Calcutta", "kolkata"],
     ["700001", "700029", "700091", "700156"],
     ["Salt Lake", "Ballygunge", "Howrah", "New Town", "Behala"]),
]

STREETS = [
    "MG Road", "Main Road", "Church Street", "Park Street", "Gandhi Nagar",
    "Nehru Street", "Station Road", "Lake View Road", "Temple Street",
    "Green Colony", "Rose Garden Street", "Hill View Road", "Market Road",
    "Anna Nagar", "Jubilee Road", "Sunrise Colony", "Banjara Street",
    "Vidya Nagar", "Sector 5 Road", "Ring Road",
]

DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "rediffmail.com", "hotmail.com"]


# ----------------------------------------------------------------------
# 2. Build one clean person ("entity")
# ----------------------------------------------------------------------

def make_phone():
    """Indian mobile numbers start with 6-9 and have 10 digits."""
    return rng.choice("6789") + "".join(rng.choice("0123456789") for _ in range(9))


def make_email(first, last):
    f, l = first.lower(), last.lower()
    pattern = rng.choice([
        f"{f}.{l}", f"{f}{l}", f"{f}_{l}", f"{f}{l}{rng.randint(1, 99)}",
        f"{f[0]}{l}", f"{f}.{l}{rng.randint(1, 999)}",
    ])
    return f"{pattern}@{rng.choice(DOMAINS)}"


def make_address(areas):
    house = rng.randint(1, 250)
    street = rng.choice(STREETS)
    area = rng.choice(areas)
    return f"{house} {street} {area}"


def make_entity(entity_id, first=None, last=None, city_row=None):
    first = first or rng.choice(FIRST_NAMES)
    last = last or rng.choice(LAST_NAMES)
    city, _, pins, areas = city_row or rng.choice(CITIES)
    return {
        "entity_id": entity_id,
        "name": f"{first} {last}",
        "email": make_email(first, last),
        "phone": make_phone(),
        "address": make_address(areas),
        "city": city,
        "pincode": rng.choice(pins),
    }


# ----------------------------------------------------------------------
# 3. Variation functions - these create the "messy" duplicates
# ----------------------------------------------------------------------

def vary_name(name):
    first, last = name.split(" ", 1)
    choice = rng.choice([
        "upper", "lower", "spaces", "initial", "initial_dot",
        "swap_order", "typo", "title", "same",
    ])
    if choice == "upper":
        return name.upper()
    if choice == "lower":
        return name.lower()
    if choice == "spaces":
        return f"  {first}   {last} "
    if choice == "initial":
        return f"{first} {last[0]}"
    if choice == "initial_dot":
        return f"{first} {last[0]}."
    if choice == "swap_order":
        return f"{last} {first}"
    if choice == "typo":                       # swap two adjacent letters
        i = rng.randrange(1, len(first) - 1) if len(first) > 3 else 1
        t = list(first)
        t[i], t[i - 1] = t[i - 1], t[i]
        return f"{''.join(t)} {last}"
    if choice == "title":
        return f"{rng.choice(['Mr.', 'Ms.', 'Mr', 'Dr.'])} {name}"
    return name


def vary_email(email):
    local, domain = email.split("@")
    choice = rng.choice(["nodots", "upper", "plus_tag", "same", "same", "swap_domain"])
    if choice == "nodots":
        return f"{local.replace('.', '').replace('_', '')}@{domain}"
    if choice == "upper":
        return email.upper()
    if choice == "plus_tag":
        return f"{local}+shop@{domain}"
    if choice == "swap_domain" and domain == "gmail.com":
        return f"{local}@googlemail.com"
    return email


def vary_phone(phone):
    choice = rng.choice([
        "plus91", "plus91_dash", "zero", "spaced", "brackets", "same", "same",
    ])
    if choice == "plus91":
        return f"+91 {phone}"
    if choice == "plus91_dash":
        return f"+91-{phone[:5]}-{phone[5:]}"
    if choice == "zero":
        return f"0{phone}"
    if choice == "spaced":
        return f"{phone[:5]} {phone[5:]}"
    if choice == "brackets":
        return f"(+91) {phone}"
    return phone


ABBREVIATIONS = {
    "Road": "Rd", "Street": "St", "Nagar": "Ngr",
    "Colony": "Cly", "Garden": "Gdn", "Sector": "Sec",
}


def vary_address(address):
    choice = rng.choice(["abbrev", "commas", "upper", "lower", "dots", "same"])
    if choice == "abbrev":
        for full, short in ABBREVIATIONS.items():
            address = address.replace(full, short)
        return address
    if choice == "commas":
        parts = address.split(" ")
        return f"{parts[0]}, {' '.join(parts[1:-1])}, {parts[-1]}"
    if choice == "upper":
        return address.upper()
    if choice == "lower":
        return address.lower()
    if choice == "dots":                       # "MG Road" -> "M.G. Road"
        return address.replace("MG", "M.G.")
    return address


def vary_city(canonical):
    for city, variants, _, _ in CITIES:
        if city == canonical:
            return rng.choice(variants)
    return canonical


def vary_pincode(pincode):
    choice = rng.choice(["same", "same", "same", "typo"])
    if choice == "typo":                       # someone mistyped the last digit
        return pincode[:-1] + rng.choice("0123456789")
    return pincode


def make_duplicate(entity, exact=False):
    """Return a messy copy of `entity`. If exact=True, copy it as-is."""
    dup = dict(entity)
    if exact:
        return dup
    dup["name"] = vary_name(entity["name"])
    dup["email"] = vary_email(entity["email"])
    dup["phone"] = vary_phone(entity["phone"])
    dup["address"] = vary_address(entity["address"])
    dup["city"] = vary_city(entity["city"])
    dup["pincode"] = vary_pincode(entity["pincode"])

    # Identifier drift.
    # Without this the dataset is unrealistically easy: every duplicate
    # would keep a perfectly matching phone AND email, so a single exact
    # rule would solve the whole problem and the evaluation would be
    # meaningless. In reality people change their number, re-register with
    # a new mailbox, or the field was simply never recorded.
    first, last = entity["name"].split(" ", 1)
    if rng.random() < 0.20:
        dup["phone"] = "" if rng.random() < 0.5 else make_phone()
    if rng.random() < 0.20:
        dup["email"] = "" if rng.random() < 0.4 else make_email(first, last)
    return dup


# ----------------------------------------------------------------------
# 4. Missing values
# ----------------------------------------------------------------------

def punch_holes(record):
    """Real datasets have blanks. Drop at most one field per record."""
    if rng.random() < 0.09:
        field = rng.choice(["email", "phone", "address", "pincode"])
        record[field] = ""
    return record


# ----------------------------------------------------------------------
# 5. Assemble the dataset
# ----------------------------------------------------------------------

def build():
    entities = [make_entity(i) for i in range(N_ENTITIES)]
    records = [dict(e) for e in entities]

    # --- duplicates -------------------------------------------------
    to_duplicate = rng.sample(entities, N_DUPLICATED)
    for i, entity in enumerate(to_duplicate):
        n_copies = rng.choices([1, 1, 1, 2, 3], weights=[5, 5, 5, 3, 1])[0]
        for c in range(n_copies):
            # the first ~18 copies are byte-identical so that plain
            # pandas .duplicated() has something to find
            exact = i < 18 and c == 0
            records.append(make_duplicate(entity, exact=exact))

    # --- confusable (different people, similar look) ----------------
    next_id = N_ENTITIES
    for _ in range(N_CONFUSABLE):
        base = rng.choice(entities)
        first, last = base["name"].split(" ", 1)
        city_row = next(c for c in CITIES if c[0] == base["city"])
        twin = make_entity(next_id, first=first, last=last, city_row=city_row)
        # same full name, same city -> but different email/phone/address
        if rng.random() < 0.4:
            twin["address"] = base["address"]        # siblings at one address
        records.append(twin)
        entities.append(twin)
        next_id += 1

    # --- shuffle, add missing values, assign record ids -------------
    rng.shuffle(records)
    records = [punch_holes(r) for r in records]
    for i, r in enumerate(records, start=1):
        r["record_id"] = f"R{i:04d}"

    df = pd.DataFrame(records)[
        ["record_id", "name", "email", "phone", "address", "city", "pincode", "entity_id"]
    ]

    # --- ground truth ----------------------------------------------
    # every pair of records sharing an entity_id is a true duplicate
    truth = []
    for _, group in df.groupby("entity_id"):
        ids = sorted(group["record_id"])
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                truth.append({"record_id_1": ids[a], "record_id_2": ids[b], "is_duplicate": 1})

    # a handful of explicitly labelled hard negatives: same name, different person
    by_name = df.assign(key=df["name"].str.lower().str.strip()).groupby("key")
    hard_negatives = []
    for _, group in by_name:
        ids = sorted(group[["record_id", "entity_id"]].values.tolist())
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                if ids[a][1] != ids[b][1]:
                    hard_negatives.append(
                        {"record_id_1": ids[a][0], "record_id_2": ids[b][0], "is_duplicate": 0}
                    )
    truth_df = pd.DataFrame(truth + hard_negatives)

    # entity_id was only a scaffolding column - the detector must not see it
    df = df.drop(columns=["entity_id"])

    df.to_csv(DATA_DIR / "sample_data.csv", index=False)
    truth_df.to_csv(DATA_DIR / "ground_truth.csv", index=False)

    print(f"records            : {len(df)}")
    print(f"true duplicate pairs: {len(truth)}")
    print(f"hard negative pairs : {len(hard_negatives)}")
    print(f"blank cells        : {(df == '').sum().sum()}")
    print(f"exact duplicate rows: {df.drop(columns=['record_id']).duplicated().sum()}")


if __name__ == "__main__":
    build()
