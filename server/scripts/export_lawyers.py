#!/usr/bin/env python3
"""
One-off: export the Indian lawyer directory from a dev database to
app/data/lawyers.json (the fixture init_db loads), correcting each row's
"City, State" location. The generated data paired cities with random states;
the city -> state table below is the source of truth. Cities missing from it
are exported with the city only rather than a wrong state.

Usage (from server/):  python scripts/export_lawyers.py
"""

import json
import os
import sys
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(_SERVER_DIR))
os.chdir(_SERVER_DIR)

from sqlmodel import Session, select

from app.db.engine import get_engine
from app.db.models import Lawyer

_STATE_CITIES = {
    "Maharashtra": "Aurangabad,Satara,Nanded,Jalna,Kolhapur,Panvel,Ichalkaranji,Nagpur,Mumbai,Bhiwandi,Ulhasnagar,Akola,Pune,Amravati,Jalgaon,Solapur,Parbhani,Ahmednagar,Pimpri-Chinchwad,Kalyan-Dombivli,Latur,Dhule,Navi Mumbai,Vasai-Virar,Mira-Bhayandar,Bhusawal,Chandrapur,Thane,Nashik,Malegaon,Sangli-Miraj & Kupwad",
    "Uttar Pradesh": "Bulandshahr,Meerut,Muzaffarnagar,Agra,Shahjahanpur,Amroha,Unnao,Rampur,Moradabad,Fatehpur,Ghaziabad,Saharanpur,Gorakhpur,Noida,Mirzapur,Bareilly,Raebareli,Sambhal,Varanasi,Hapur,Lucknow,Jaunpur,Allahabad,Firozabad,Jhansi,Etawah,Kanpur,Orai,Aligarh,Ballia,Mau,Farrukhabad,Loni,Mathura,Bahraich",
    "Andhra Pradesh": "Nellore,Proddatur,Visakhapatnam,Adoni,Kurnool,Bhimavaram,Kakinada,Machilipatnam,Kadapa,Nandyal,Narasaraopet,Anantapur,Dharmavaram,Madanapalle,Rajahmundry,Amaravati,Kavali,Vijayawada,Hospet,Guntur,Tadepalligudem,Hindupur,Anantapuram,Eluru,Ongole,Chittoor,Tirupati,Guntakal,Vijayanagaram,Srikakulam,Gudivada,Tenali,Miryalaguda",
    "Karnataka": "Davanagere,Hubli–Dharwad,Bangalore,Mysore,Bellary,Raichur,Tumkur,Gulbarga,Shimoga,Udupi,Belgaum,Bidar,Hospet,Bijapur",
    "Assam": "Tinsukia,Jorhat,Tezpur,Silchar,Guwahati,Nagaon,Bongaigaon,Dibrugarh",
    "Punjab": "Patiala,Phagwara,Amritsar,Ludhiana,Jalandhar,Bathinda",
    "West Bengal": "Howrah,Uluberia,Haldia,Kharagpur,Maheshtala,Asansol,Kulti,Siliguri,Rajpur Sonarpur,Bhatpara,Naihati,Bally,Raiganj,Barasat,Durgapur,Kolkata,Malda,Bidhannagar,Berhampore,Panihati,Baranagar,Chinsurah,North Dumdum,Madhyamgram,South Dumdum,Bardhaman",
    "Telangana": "Secunderabad,Karimnagar,Warangal,Khammam,Mahbubnagar,Hyderabad,Suryapet,Nizamabad,Ramagundam,Miryalaguda",
    "Bihar": "Buxar,Muzaffarpur,Chapra,Danapur,Saharsa,Siwan,Dehri,Darbhanga,Bhagalpur,Arrah,Katihar,Bihar Sharif,Kishanganj,Patna,Bettiah,Munger,Hajipur,Jehanabad,Motihari,Gaya,Begusarai,Sasaram",
    "Jharkhand": "Giridih,Phusro,Hazaribagh,Ranchi,Deoghar,Jamshedpur,Ramgarh,Medininagar,Dhanbad,Bokaro",
    "Tamil Nadu": "Tiruchirappalli,Karaikudi,Pallavaram,Dindigul,Hosur,Vellore,Nagercoil,Thanjavur,Salem,Tiruppur,Avadi,Ozhukarai,Tirunelveli,Ambattur,Erode,Pudukkottai,Thoothukudi,Coimbatore,Tiruvottiyur,Kumbakonam,Chennai,Madurai",
    "Jammu and Kashmir": "Srinagar,Jammu",
    "Madhya Pradesh": "Satna,Ujjain,Ratlam,Bhopal,Katni,Guna,Sagar,Jabalpur,Shivpuri,Dewas,Rewa,Indore,Khandwa,Morena,Gwalior,Burhanpur,Bhind,Singrauli",
    "Gujarat": "Vadodara,Gandhidham,Morbi,Surendranagar Dudhrej,Rajkot,Nadiad,Mehsana,Anand,Bhavnagar,Junagadh,Surat,Ahmedabad,Jamnagar,Gandhinagar",
    "Himachal Pradesh": "Shimla",
    "Rajasthan": "Udaipur,Ajmer,Alwar,Jaipur,Sikar,Jodhpur,Bikaner,Bharatpur,Kota,Bhilwara,Pali",
    "Haryana": "Gurgaon,Faridabad,Panchkula,Sirsa,Rohtak,Yamunanagar,Bhiwani,Ambala,Panipat,Sonipat,Karnal",
    "Chhattisgarh": "Durg,Raipur,Bhilai,Korba,Bilaspur",
    "Manipur": "Imphal",
    "Delhi": "Delhi,New Delhi,Kirari Suleman Nagar,Nangloi Jat,Karawal Nagar,Sultan Pur Majra,Bhalswa Jahangir Pur",
    "Sikkim": "Gangtok",
    "Kerala": "Kollam,Kottayam,Kochi,Thiruvananthapuram,Alappuzha,Thrissur",
    "Odisha": "Bhubaneswar,Berhampur,Cuttack,Rourkela,Raurkela Industrial Township,Gopalpur",
    "Tripura": "Agartala",
    "Chandigarh": "Chandigarh",
    "Puducherry": "Pondicherry",
    "Mizoram": "Aizawl",
    "Uttarakhand": "Haridwar,Dehradun",
}
CITY_TO_STATE = {c.strip(): st for st, cities in _STATE_CITIES.items() for c in cities.split(",")}


def fixed_location(location: str) -> str:
    city = location.split(",")[0].strip()
    state = CITY_TO_STATE.get(city)
    return f"{city}, {state}" if state else city


def main() -> None:
    with Session(get_engine()) as session:
        rows = session.exec(
            select(Lawyer).where(Lawyer.id.not_in(["1", "2", "3", "4", "5"])).order_by(Lawyer.id)
        ).all()
        out = []
        unmapped = set()
        for lw in rows:
            d = lw.to_dict()
            d["location"] = fixed_location(lw.location)
            if "," not in d["location"]:
                unmapped.add(d["location"])
            out.append(d)
    path = _SERVER_DIR / "app" / "data" / "lawyers.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Exported {len(out)} lawyers to {path}; {len(unmapped)} cities left without a state: {sorted(unmapped)}")


if __name__ == "__main__":
    main()
