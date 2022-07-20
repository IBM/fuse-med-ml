"""
(C) Copyright 2021 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on June 30, 2021

"""

import os
import requests
import zipfile
import io
import wget
from zipfile import ZipFile

import logging

GOLDEN_MEMBERS = [
    "ISIC_0072637",
    "ISIC_0072638",
    "ISIC_0072639",
    "ISIC_0072640",
    "ISIC_0072641",
    "ISIC_0072642",
    "ISIC_0072646",
    "ISIC_0072647",
    "ISIC_0072648",
    "ISIC_0072649",
    "ISIC_0072650",
    "ISIC_0072651",
    "ISIC_0072652",
    "ISIC_0072653",
    "ISIC_0072654",
    "ISIC_0072656",
    "ISIC_0072657",
    "ISIC_0072658",
    "ISIC_0072659",
    "ISIC_0072660",
    "ISIC_0072661",
    "ISIC_0072662",
    "ISIC_0072663",
    "ISIC_0072665",
    "ISIC_0072666",
    "ISIC_0072668",
    "ISIC_0072670",
    "ISIC_0072673",
    "ISIC_0072675",
    "ISIC_0072676",
    "ISIC_0072679",
    "ISIC_0072683",
    "ISIC_0072687",
    "ISIC_0072688",
    "ISIC_0072690",
    "ISIC_0072691",
    "ISIC_0072693",
    "ISIC_0072694",
    "ISIC_0072697",
    "ISIC_0072698",
    "ISIC_0072699",
    "ISIC_0072701",
    "ISIC_0072703",
    "ISIC_0072708",
    "ISIC_0072709",
    "ISIC_0072710",
    "ISIC_0072713",
    "ISIC_0072714",
    "ISIC_0072715",
    "ISIC_0072716",
    "ISIC_0072717",
    "ISIC_0072723",
    "ISIC_0072726",
    "ISIC_0072727",
    "ISIC_0072729",
    "ISIC_0072730",
    "ISIC_0072731",
    "ISIC_0072732",
    "ISIC_0072733",
    "ISIC_0072735",
    "ISIC_0072737",
    "ISIC_0072738",
    "ISIC_0072739",
    "ISIC_0072740",
    "ISIC_0072741",
    "ISIC_0072742",
    "ISIC_0072744",
    "ISIC_0072745",
    "ISIC_0072747",
    "ISIC_0072748",
    "ISIC_0072749",
    "ISIC_0072750",
    "ISIC_0072752",
    "ISIC_0072754",
    "ISIC_0072755",
    "ISIC_0072759",
    "ISIC_0072760",
    "ISIC_0072761",
    "ISIC_0072764",
    "ISIC_0072767",
    "ISIC_0072768",
    "ISIC_0072770",
    "ISIC_0072771",
    "ISIC_0072772",
    "ISIC_0072773",
    "ISIC_0072774",
    "ISIC_0072775",
    "ISIC_0072776",
    "ISIC_0072777",
    "ISIC_0072778",
    "ISIC_0072781",
    "ISIC_0072782",
    "ISIC_0072784",
    "ISIC_0072785",
    "ISIC_0072786",
    "ISIC_0072788",
    "ISIC_0072789",
    "ISIC_0072792",
    "ISIC_0072795",
    "ISIC_0072796",
    "ISIC_0072798",
    "ISIC_0072799",
    "ISIC_0072802",
    "ISIC_0072805",
    "ISIC_0072806",
    "ISIC_0072807",
    "ISIC_0072809",
    "ISIC_0072810",
    "ISIC_0072813",
    "ISIC_0072814",
    "ISIC_0072815",
    "ISIC_0072818",
    "ISIC_0072820",
    "ISIC_0072821",
    "ISIC_0072822",
    "ISIC_0072824",
    "ISIC_0072825",
    "ISIC_0072826",
    "ISIC_0072827",
    "ISIC_0072828",
    "ISIC_0072829",
    "ISIC_0072830",
    "ISIC_0072833",
    "ISIC_0072834",
    "ISIC_0072835",
    "ISIC_0072836",
    "ISIC_0072837",
    "ISIC_0072838",
    "ISIC_0072839",
    "ISIC_0072841",
    "ISIC_0072842",
    "ISIC_0072843",
    "ISIC_0072844",
    "ISIC_0072846",
    "ISIC_0072847",
    "ISIC_0072848",
    "ISIC_0072850",
    "ISIC_0072851",
    "ISIC_0072852",
    "ISIC_0072853",
    "ISIC_0072854",
    "ISIC_0072855",
    "ISIC_0072856",
    "ISIC_0072857",
    "ISIC_0072858",
    "ISIC_0072859",
    "ISIC_0072860",
    "ISIC_0072861",
    "ISIC_0072864",
    "ISIC_0072865",
    "ISIC_0072866",
    "ISIC_0072868",
    "ISIC_0072869",
    "ISIC_0072870",
    "ISIC_0072871",
    "ISIC_0072872",
    "ISIC_0072874",
    "ISIC_0072876",
    "ISIC_0072877",
    "ISIC_0072878",
    "ISIC_0072880",
    "ISIC_0072881",
    "ISIC_0072882",
    "ISIC_0072885",
    "ISIC_0072887",
    "ISIC_0072888",
    "ISIC_0072889",
    "ISIC_0072891",
    "ISIC_0072892",
    "ISIC_0072893",
    "ISIC_0072894",
    "ISIC_0072895",
    "ISIC_0072896",
    "ISIC_0072897",
    "ISIC_0072898",
    "ISIC_0072900",
    "ISIC_0072901",
    "ISIC_0072902",
    "ISIC_0072904",
    "ISIC_0072905",
    "ISIC_0072907",
    "ISIC_0072909",
    "ISIC_0072910",
    "ISIC_0072911",
    "ISIC_0072914",
    "ISIC_0072916",
    "ISIC_0072917",
    "ISIC_0072918",
    "ISIC_0072919",
    "ISIC_0072923",
    "ISIC_0072924",
    "ISIC_0072926",
    "ISIC_0072928",
    "ISIC_0072929",
    "ISIC_0072931",
    "ISIC_0072932",
    "ISIC_0072933",
    "ISIC_0072935",
    "ISIC_0072937",
    "ISIC_0072938",
    "ISIC_0072939",
    "ISIC_0072940",
    "ISIC_0072941",
    "ISIC_0072942",
    "ISIC_0072943",
    "ISIC_0072951",
    "ISIC_0072953",
    "ISIC_0072957",
    "ISIC_0072960",
    "ISIC_0072962",
    "ISIC_0072964",
    "ISIC_0072965",
    "ISIC_0072966",
    "ISIC_0072969",
    "ISIC_0072970",
    "ISIC_0072972",
    "ISIC_0072974",
    "ISIC_0072975",
    "ISIC_0072976",
    "ISIC_0072977",
    "ISIC_0072978",
    "ISIC_0072983",
    "ISIC_0072985",
    "ISIC_0072986",
    "ISIC_0072988",
    "ISIC_0072989",
    "ISIC_0072990",
    "ISIC_0072991",
    "ISIC_0072992",
    "ISIC_0072993",
    "ISIC_0072994",
    "ISIC_0072997",
    "ISIC_0072998",
    "ISIC_0072999",
    "ISIC_0073001",
    "ISIC_0073002",
    "ISIC_0073003",
    "ISIC_0073004",
    "ISIC_0073005",
    "ISIC_0073008",
    "ISIC_0073009",
    "ISIC_0073010",
    "ISIC_0073011",
    "ISIC_0073012",
    "ISIC_0073015",
    "ISIC_0073016",
    "ISIC_0073018",
    "ISIC_0073019",
    "ISIC_0073020",
    "ISIC_0073021",
    "ISIC_0073022",
    "ISIC_0073023",
    "ISIC_0073025",
    "ISIC_0073027",
    "ISIC_0073028",
    "ISIC_0073030",
    "ISIC_0073031",
    "ISIC_0073032",
    "ISIC_0073033",
    "ISIC_0073035",
    "ISIC_0073036",
    "ISIC_0073037",
    "ISIC_0073038",
    "ISIC_0073039",
    "ISIC_0073043",
    "ISIC_0073044",
    "ISIC_0073045",
    "ISIC_0073047",
    "ISIC_0073048",
    "ISIC_0073049",
    "ISIC_0073050",
    "ISIC_0073054",
    "ISIC_0073055",
    "ISIC_0073056",
    "ISIC_0073058",
    "ISIC_0073059",
    "ISIC_0073060",
    "ISIC_0073061",
    "ISIC_0073063",
    "ISIC_0073065",
    "ISIC_0073066",
    "ISIC_0073068",
    "ISIC_0073069",
    "ISIC_0073070",
    "ISIC_0073071",
    "ISIC_0073072",
    "ISIC_0073075",
    "ISIC_0073077",
    "ISIC_0073079",
    "ISIC_0073080",
    "ISIC_0073081",
    "ISIC_0073082",
    "ISIC_0073086",
    "ISIC_0073088",
    "ISIC_0073091",
    "ISIC_0073092",
    "ISIC_0073097",
    "ISIC_0073099",
    "ISIC_0073100",
    "ISIC_0073101",
    "ISIC_0073102",
    "ISIC_0073104",
    "ISIC_0073105",
    "ISIC_0073108",
    "ISIC_0073109",
    "ISIC_0073110",
    "ISIC_0073111",
    "ISIC_0073112",
    "ISIC_0073113",
    "ISIC_0073115",
    "ISIC_0073116",
    "ISIC_0073119",
    "ISIC_0073122",
    "ISIC_0073123",
    "ISIC_0073125",
    "ISIC_0073126",
    "ISIC_0073127",
    "ISIC_0073128",
    "ISIC_0073130",
    "ISIC_0073133",
    "ISIC_0073135",
    "ISIC_0073136",
    "ISIC_0073137",
    "ISIC_0073138",
    "ISIC_0073140",
    "ISIC_0073141",
    "ISIC_0073142",
    "ISIC_0073143",
    "ISIC_0073144",
    "ISIC_0073146",
    "ISIC_0073147",
    "ISIC_0073148",
    "ISIC_0073149",
    "ISIC_0073150",
    "ISIC_0073151",
    "ISIC_0073153",
    "ISIC_0073154",
    "ISIC_0073155",
    "ISIC_0073156",
    "ISIC_0073157",
    "ISIC_0073159",
    "ISIC_0073161",
    "ISIC_0073164",
    "ISIC_0073168",
    "ISIC_0073169",
    "ISIC_0073170",
    "ISIC_0073171",
    "ISIC_0073172",
    "ISIC_0073173",
    "ISIC_0073175",
    "ISIC_0073181",
    "ISIC_0073182",
    "ISIC_0073183",
    "ISIC_0073184",
    "ISIC_0073187",
    "ISIC_0073189",
    "ISIC_0073193",
    "ISIC_0073194",
    "ISIC_0073195",
    "ISIC_0073196",
    "ISIC_0073198",
    "ISIC_0073199",
    "ISIC_0073200",
    "ISIC_0073201",
    "ISIC_0073202",
    "ISIC_0073203",
    "ISIC_0073205",
    "ISIC_0073207",
    "ISIC_0073208",
    "ISIC_0073209",
    "ISIC_0073210",
    "ISIC_0073212",
    "ISIC_0073214",
    "ISIC_0073215",
    "ISIC_0073218",
    "ISIC_0073219",
    "ISIC_0073220",
    "ISIC_0073221",
    "ISIC_0073222",
    "ISIC_0073223",
    "ISIC_0073224",
    "ISIC_0073225",
    "ISIC_0073227",
    "ISIC_0073228",
    "ISIC_0073229",
    "ISIC_0073231",
    "ISIC_0073232",
    "ISIC_0073235",
    "ISIC_0073237",
    "ISIC_0073238",
    "ISIC_0073240",
    "ISIC_0073241",
    "ISIC_0073244",
    "ISIC_0073245",
    "ISIC_0073246",
    "ISIC_0073247",
    "ISIC_0073248",
    "ISIC_0073249",
    "ISIC_0073251",
    "ISIC_0073254",
]


def download_and_extract_isic(root_data: str = "data", golden_only: bool = False):
    """
    Download images and metadata from ISIC challenge
    :param root_data: path where data should be located
    """
    lgr = logging.getLogger("Fuse")

    path = os.path.join(root_data, "ISIC2019/ISIC_2019_Training_Input")
    print(f"Training Input Path: {os.path.abspath(path)}")
    if not os.path.exists(path):
        lgr.info("\nExtract ISIC-2019 training input ... (this may take a few minutes)")

        url = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"
        wget.download(url, ".")

        if golden_only:
            members = [
                os.path.join("ISIC_2019_Training_Input", m + ".jpg")
                for m in GOLDEN_MEMBERS
            ]
        else:
            members = None
        with ZipFile("ISIC_2019_Training_Input.zip", "r") as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(path=os.path.join(root_data, "ISIC2019"), members=members)

        lgr.info("Extracting ISIC-2019 training input: done")

    path = os.path.join(root_data, "ISIC2019/ISIC_2019_Training_Metadata.csv")
    print(f"Training Metadata Path: {os.path.abspath(path)}")
    if not os.path.exists(path):
        lgr.info(
            "\nExtract ISIC-2019 training metadata ... (this may take a few minutes)"
        )

        url = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv"
        wget.download(url, path)

        lgr.info("Extracting ISIC-2019 training metadata: done")

    path = os.path.join(root_data, "ISIC2019/ISIC_2019_Training_GroundTruth.csv")
    print(f"Training Metadata Path: {os.path.abspath(path)}")
    if not os.path.exists(path):
        lgr.info("\nExtract ISIC-2019 training gt ... (this may take a few minutes)")

        url = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"
        wget.download(url, path)

        lgr.info("Extracting ISIC-2019 training gt: done")
