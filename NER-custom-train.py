import spacy

nlp = spacy.load("en_core_web_sm")
text = ("Main Customs Office Potsdam Day of Arrival in Berlin SXF 19/09/2023 SG C - KR 1 - KEFR - KE 11 Flight Number MH1234 Berlin-Airport Tegel Reference Number (loss report already done) Customs Declaration For lost / forwarded Baggage Each arriving traveler or responsible fellow passenger must provide the following information for lost/forwarded baggage. 1. Traveler Name: JOHN DOE First name: JOHN Fellow Nr. of travelers under travelers: the age of 15/17: 0 City: BERLIN Street / Number: Lychener Str. 46 Country: BERLIN Zip code: 54580 2. Information about goods In my lost / forwarded baggage There are no goods from a non-EU country in my lost / forwarded baggage. X My lost / forwarded baggage contains the following goods from non-EU countries: Goods Description of articles/Amount Value In Euro Tobacco products Alcohol and beverages containing alcohol Medical products XANNAX 20 XI Other goods 3. Information about my other baggage (Incl. carry-on baggage) which I'm carrying with me right now . There are no goods from a non-EU country in my other Baggage (incl. carry-on baggage). X My other Baggage (incl. carry-on baggage) contains the following goods from non-EU countries Goods Description of articles/Amount Value In Euro Tobacco products Alcohol and beverages containing alcohol 4 Medical products IVERMECTIN 50 Other goods I am aware that a false, Inaccurate or Incomplete declaration may lead to legal consequences.")
doc = nlp(text)

train = [
    ("Form I Tag a c a INDIAN CUSTOMS DECLARATION FORM (Please see important information given below before filling this Form) 1. Name of the JANE DOE Passenger 2. Passport Number AA12345678 3. Nationality MALAYSIAN 4. Date of Arrival 18/09/2023DD/MM/YVY) 5. Flight No. AK123 6. Number of Baggages 2 ...-........ 7. Country from where coming MALAYSIA (including hand baggages) 8. Countries visited in last six days NIL 9. Total value of dutiable goods being imported (Rs.) 10. Are you bringing the following items into India? (please tick Yes or No) (i) Prohibited Articles Yes (11) Gold jewellery (over Free Allowance) Yes (11) Gold Bullion No (iv) Meat and meat products/dairy products/fish/poultry products Yes (v) Seeds/plants/seeds/fruits/flowers/other planting material Yes (/ No (vi) Satellite phone Yes (vii) Indian currency exceeding Rs. 10,000/- Yes / No (viii) Foreign currency notes exceed US $ 5,000 or equivalent No (ix) Aggregate value of foreign exchange including currency exceeds US S 10,000 or equivalent. Yes / No Please report to Customs Officer at the Red Channel counter in case answer to any of the above question is 'Yes'. Signature of Passenger ...............................",{"entities":[(136,144,"name"),(229,239,"date"),(264,271,"flight_number")]}),
    ("Main Customs Office Potsdam Day of Arrival in Berlin SXF 19/09/2023 SG C - KR 1 - KEFR - KE 11 Flight Number MH1234 Berlin-Airport Tegel Reference Number (loss report already done) Customs Declaration For lost / forwarded Baggage Each arriving traveler or responsible fellow passenger must provide the following information for lost/forwarded baggage. 1. Traveler Name: JOHN DOE First name: JOHN Fellow Nr. of travelers under travelers: the age of 15/17: 0 City: BERLIN Street / Number: Lychener Str. 46 Country: BERLIN Zip code: 54580 2. Information about goods In my lost / forwarded baggage There are no goods from a non-EU country in my lost / forwarded baggage. X My lost / forwarded baggage contains the following goods from non-EU countries: Goods Description of articles/Amount Value In Euro Tobacco products Alcohol and beverages containing alcohol Medical products XANNAX 20 XI Other goods 3. Information about my other baggage (Incl. carry-on baggage) which I'm carrying with me right now . There are no goods from a non-EU country in my other Baggage (incl. carry-on baggage). X My other Baggage (incl. carry-on baggage) contains the following goods from non-EU countries Goods Description of articles/Amount Value In Euro Tobacco products Alcohol and beverages containing alcohol 4 Medical products IVERMECTIN 50 Other goods I am aware that a false, Inaccurate or Incomplete declaration may lead to legal consequences.",{"entities":[(57,67,"date"),(370,378,"name"),(109,115,"flight_number"),(876,882,"goods"),(1314,1324,"goods")]}),
]

import pandas as pd
import os
from tqdm import tqdm
from spacy.tokens import DocBin

# db = DocBin() # create a DocBin object

# for text, annot in tqdm(train): # data in previous format
#     doc = nlp.make_doc(text) # create doc object from text
#     ents = []
#     for start, end, label in annot["entities"]: # add character indexes
#         span = doc.char_span(start, end, label=label, alignment_mode="contract")
#         if span is None:
#             print("Skipping entity")
#         else:
#             ents.append(span)
#     doc.ents = ents # label the text with the ents
#     db.add(doc)

# db.to_disk("./train.spacy") # save the docbin object

# python3.8 -m spacy init fill-config base_config.cfg config.cfg
# python3.8 -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./train.spacy

nlp1 = spacy.load(r"./output/model-best") #load the best model
doc = nlp1("Main Customs Office Potsdam Day of Arrival in Berlin SXF 19/09/2023 SG C - KR 1 - KEFR - KE 11 Flight Number MH1234 Berlin-Airport Tegel Reference Number (loss report already done) Customs Declaration For lost / forwarded Baggage Each arriving traveler or responsible fellow passenger must provide the following information for lost/forwarded baggage. 1. Traveler Name: JOHN DOE First name: JOHN Fellow Nr. of travelers under travelers: the age of 15/17: 0 City: BERLIN Street / Number: Lychener Str. 46 Country: BERLIN Zip code: 54580 2. Information about goods In my lost / forwarded baggage There are no goods from a non-EU country in my lost / forwarded baggage. X My lost / forwarded baggage contains the following goods from non-EU countries: Goods Description of articles/Amount Value In Euro Tobacco products Alcohol and beverages containing alcohol Medical products XANNAX 20 XI Other goods 3. Information about my other baggage (Incl. carry-on baggage) which I'm carrying with me right now . There are no goods from a non-EU country in my other Baggage (incl. carry-on baggage). X My other Baggage (incl. carry-on baggage) contains the following goods from non-EU countries Goods Description of articles/Amount Value In Euro Tobacco products Alcohol and beverages containing alcohol 4 Medical products IVERMECTIN 50 Other goods I am aware that a false, Inaccurate or Incomplete declaration may lead to legal consequences") # input sample text

print("Custom Entities Found:")
for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")