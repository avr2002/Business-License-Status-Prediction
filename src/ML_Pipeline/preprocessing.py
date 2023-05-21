import re
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from ML_Pipeline.constants import CATEGORICAL_FEATURES, PREDICTORS

import warnings
warnings.filterwarnings('ignore')


def similar(a, b):
    """
     Gives out a similarity score b/w two strings b/w [0, 1], 
     As a rule of thumb, ratio value over 0.6 means the sequences are close matches
    """
    return SequenceMatcher(None, a, b).ratio()

def convert_zipcodes_to_int(x):
    """
    Converts the ZIP_CODE column in the data to integer
    And also replaces the string values with -1
    """
    x = str(x)
    if x.endswith('.0'):
        return int(x.split('.')[0])
    if x.isnumeric():
        return int(x)
    if x=='-1':
        return -1
    
    return -1

def data_cleaning(data):
    """
    **SEE THE MAIN NOTEBOOK FOR LOGIC BEHIND THE FEATURE ENGINEERING STEPS**

    Here are the following pre-processing done to the data:
        1. New Feature `LEGAL_BUSINESS_NAME_MATCH` is made using `LEGAL_NAME` & `DOING_BUSINESS_AS_NAME`
        
        2. Combining diff. categories into one in `LICENSE_DESCRIPTION` to shoten the num. of categories
            Reduced from `106` total categories to `46`
        
        3. New Feature `BUSINESS_TYPE` is created using `LEGAL_NAME` & `DOING_BUSINESS_AS_NAME`
        
        4. New Feature `IS_ZIP_CODE_PRESENT` is created using `ZIP_CODE`
        
        5. `SSA` & `APPLICATION_REQUIREMENTS_COMPLETE` features has been modified

        6. `STATE` with less records has merged to `Other_States` categories.

        7. New Feature `HAS_LICENSE_STATUS_CHANGED` is created using `LICENSE_TERM_START_DATE` & `LICENSE_STATUS_CHANGE_DATE`

        8. Following features has been selected for Modelling purposes:
            ['STATE','LICENSE_DESCRIPTION', 'APPLICATION_TYPE', 'APPLICATION_REQUIREMENTS_COMPLETE', 
            'CONDITIONAL_APPROVAL', 'SSA', 'LEGAL_BUSINESS_NAME_MATCH', 'BUSINESS_TYPE', 
            'IS_ZIP_CODE_PRESENT', 'HAS_LICENSE_STATUS_CHANGED', 'LICENSE_STATUS']
    """
    # Making a new feature called LEGAL_BUSINESS_NAME_MATCH to see if LEGAL_NAME and DOING_BUSINESS_AS_NAME have some similarity or not
    data['LEGAL_BUSINESS_NAME_MATCH'] = data.apply(lambda x: similar(str(x['LEGAL_NAME']).upper(),
                                                             str(x['DOING_BUSINESS_AS_NAME']).upper()), 
                                                   axis=1)
    
    # Combining diff. categories into one in 'LICENSE_DESCRIPTION' to shoten the num. of categories
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Peddler, non-food', 'Peddler')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Peddler, non-food, special', 'Peddler')

    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Retail Food Establishment', 'Food Establishment')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Wholesale Food Establishment', 'Food Establishment')

    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Motor Vehicle Repair; Specialty(Class I)', 'Motor Vehicle')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Motor Vehicle Repair : Engine Only (Class II)', 'Motor Vehicle')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Motor Vehicle Repair: Engine/Body(Class III)', 'Motor Vehicle')

    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Day Care Center Under 2 Years', 'Day Care Center')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Day Care Center 2 - 6 Years', 'Day Care Center')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Day Care Center Under 2 and 2 - 6 Years', 'Day Care Center')

    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Repossessor Class A', 'Repossessor')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Repossessor Class B', 'Repossessor')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Repossessor Class B Employee', 'Repossessor')

    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Tire Facilty Class I (100 - 1,000 Tires)', 'Tire Facilty')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Tire Facility Class II (1,001 - 5,000 Tires)', 'Tire Facilty')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Tire Facility Class III (5,001 - More Tires)', 'Tire Facilty')

    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Expediter - Class A', 'Expediter')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Expediter - Class B', 'Expediter')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Expediter - Class B Employee', 'Expediter')

    # data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Public Place of Amusement')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Public Place of Amusement-TCC', 'Public Place of Amusement')

    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Long-Term Care Facility', 'Care Facility')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Animal Care Facility', 'Care Facility')

    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Single Room Occupancy Class I', 'Single Room Occupancy')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Single Room Occupancy Class II', 'Single Room Occupancy')

    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Itinerant Merchant, Class I', 'Itinerant Merchant')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Itinerant Merchant, Class II', 'Itinerant Merchant')

    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace("Airport Pushcart Liquor O'Hare - Class A", 
                                                                      'Airport Pushcart Liquor')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Airport Pushcart Liquor Midway - Class A', 
                                                                      'Airport Pushcart Liquor')

    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Massage Establishment', 'Massage Services')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Massage Therapist', 'Massage Services')

    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Mobile Food Dispenser','Food Dispenser')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Mobile Frozen Desserts Dispenser - Non-Motorized','Food Dispenser')

    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Junk Peddler', 'Peddler')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Peddler,food - (fruits and vegetables only) - special', 'Peddler')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace("Peddler, food (fruits and vegtables only)", 'Peddler')

    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Tobacco Vending Machine Operator', 'Tobacco')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Tobacco Dealer Wholesale', 'Tobacco')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Tobacco Vending, Individual', 'Tobacco')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace("Tobacco Sampler", 'Tobacco')

    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Motor Vehicle Services License', 'Motor Vehicle')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Kennels and Catteries', 'Pet Shop')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Retail Food Est.-Supplemental License for Dog-Friendly Areas', 'Food Establishment')

    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Riverwalk Venue Liquor License', 'Liquor License')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace("Caterer's Liquor License", 'Liquor License')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace("Caterer's Registration (Liquor)","Liquor License")
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace("Airport Pushcart Liquor", "Liquor License")
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace("Consumption on Premises - Incidental Activity", 'Liquor License')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace("Tavern", "Liquor License")

    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace("Secondhand Dealer - Children's Products", "Secondhand Dealer")
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace("Secondhand Dealer (No Valuable Objects)", "Secondhand Dealer")

    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace("Navy Pier - Mobile", "Navy Pier Vendor")
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace("Navy Pier Vendor (Non-Food)", "Navy Pier Vendor")
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace("Navy Pier - Outdoor Fixed", "Navy Pier Vendor")

    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace("Explosives", "Weapons Dealer")

    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace("Laundry, Late Hour", "Late Hour")

    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace("Pawnbroker", 'Broker')

    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace("Single Room Occupancy", "Hotel")
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace("Raffles", 'Hotel')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace("Bed-And-Breakfast Establishment", 'Hotel')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace("Assisted Living/Shared Housing Establishment", "Hotel")
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace("Vacation Rental", "Hotel")

    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace("Veterinary Hospital", "Hospital")

    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace("Outdoor Patio")

    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace("Board-Up Work", "Regulated Business License")

    # Combining categories with less records into one as "Others"
    other_categories = ['Explosives, Certificate of Fitness', 'Humane Society',
                        'Produce Merchant', 'Valet Parking Operator', 'Weapons Dealer',
                        'Wrigley Field', 'Bicycle Messenger Service', 'Performing Arts Venue',
                        'Animal Care License', 'Private Booting Operation', 'Affiliation',
                        'Guard Dog Service', 'Indoor Special Event', 'Not-For-Profit Club']
    
    for i in other_categories:
        data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace(i, "Others")
    

    # Creating a BUSINESS_TYPE feature using LEGAL_NAME/DOING_BUSINESS_AS_NAME

    # Removing dots from business names : INC. --> INC or CO. ---> CO
    data['LEGAL_NAME'] = data['LEGAL_NAME'].str.replace('.', '', regex=False)
    data['DOING_BUSINESS_AS_NAME'] = data['DOING_BUSINESS_AS_NAME'].str.replace('.', '', regex=False)

    data['BUSINESS_TYPE'] = 'PVT'  # default value 'PVT' and we will change it accordingly

    data['BUSINESS_TYPE'] = np.where(data['LEGAL_NAME'].str.contains('LLC|L.L.C',flags=re.IGNORECASE,regex=True),
                                'LLC', data['BUSINESS_TYPE'])

    co_pattern = "^(?!.*\\b(INC|LLC|LIMITED)\\b)(?=.*\\b(CO|CORP|CORPORATION)\\b).+$"
    data['BUSINESS_TYPE'] = np.where(data['LEGAL_NAME'].str.contains(co_pattern,regex=True,flags=re.IGNORECASE),
                                'CORP', data['BUSINESS_TYPE'])

    ltd_pattern = "^(?!.*\\b(INC|LLC)\\b)(?=.*\\b(LTD|LIMITED)\\b).+$"
    data['BUSINESS_TYPE'] = np.where(data['LEGAL_NAME'].str.contains(ltd_pattern,regex=True,flags=re.IGNORECASE),
                                'LTD', data['BUSINESS_TYPE'])

    data['BUSINESS_TYPE'] = np.where(data['LEGAL_NAME'].str.contains("INC|INCORPORATED",flags=re.IGNORECASE,regex=True),
                                'INC', data['BUSINESS_TYPE'])
    
    # Creating a new feature 'IS_ZIP_CODE_PRESENT'
    
    data['ZIP_CODE'].fillna(-1, inplace=True)

    # Some Zip Codes have string values in them, we'll be replacing them with -1
    data['ZIP_CODE'] = data['ZIP_CODE'].apply(lambda x: convert_zipcodes_to_int(x))

    data['IS_ZIP_CODE_PRESENT'] = data.apply(lambda x: 1 if x['ZIP_CODE']!=-1 else 0, axis=1)

    # Filling null values with -1 in 'SSA' feature
    data['SSA'].fillna(-1, inplace=True)

    # Converting 'APPLICATION_REQUIREMENTS_COMPLETE' feature to a binary feature i.e. YES/NO
    data['APPLICATION_REQUIREMENTS_COMPLETE'].fillna(-1, inplace=True)
    data['APPLICATION_REQUIREMENTS_COMPLETE'] = data.apply(lambda x: 0 if x['APPLICATION_REQUIREMENTS_COMPLETE'] == -1 
                                                               else 1, axis=1)

    # Combining STATES with less records into one as "Other_States"

    other_states = ['AR', 'DE', 'ID', 'MT', 'ON', 'NH', 'HI', 'NM', 'VT', 'AK', 'CN', 'WV',
                    'ME', 'GB', 'WY']
    for i in other_states:
        data['STATE'] = data['STATE'].replace(i, "Other_States")


    # Making a new feature which tells us, if the bussiness licence was changed or not

    data['LICENSE_TERM_START_DATE'] = data['LICENSE_TERM_START_DATE'].astype(np.datetime64)
    data['LICENSE_STATUS_CHANGE_DATE'] = data['LICENSE_STATUS_CHANGE_DATE'].astype(np.datetime64)

    # By analyzing these two columns 'LICENSE_TERM_START_DATE','LICENSE_STATUS_CHANGE_DATE'
    # We will form a new feature based called 'HAS_LICENSE_STATUS_CHANGED'

    # Index of records where license status was not changed
    # When both have null values simultaneously, we consider status not changed
    # When only 'LICENSE_STATUS_CHANGE_DATE' has null value and not the other, we consider status not changed
    status_not_changed1 = data[~data.LICENSE_TERM_START_DATE.isna() & data.LICENSE_STATUS_CHANGE_DATE.isna()].index
    status_not_changed2 = data[data.LICENSE_TERM_START_DATE.isna() & data.LICENSE_STATUS_CHANGE_DATE.isna()].index

    # Index of records where license status was not changed

    # When both do not have null values simultaneously, we consider status changed (Both feature does not contain same values when not-null)
    # When only 'LICENSE_TERM_START_DATE' has null value and not the other, we consider status changed
    status_changed1 = data[~data.LICENSE_TERM_START_DATE.isna() & ~data.LICENSE_STATUS_CHANGE_DATE.isna()].index
    status_changed2 = data[data.LICENSE_TERM_START_DATE.isna() & ~data.LICENSE_STATUS_CHANGE_DATE.isna()].index

    data['HAS_LICENSE_STATUS_CHANGED'] = 0  # (NO)
        
    for i in status_changed1:
        data.loc[i, 'HAS_LICENSE_STATUS_CHANGED'] = 1   # YES
        
    for i in status_changed2:
        data.loc[i, 'HAS_LICENSE_STATUS_CHANGED'] = 1   # YES

    # print("Data Cleaning Done")

    # Selecting the necessary columns & Saving the preprocessed data
    cols = PREDICTORS+['LICENSE_STATUS']
    
    df = data[cols]
    # df.to_csv("..\..\output\preprocessed_License_Data.csv", index=False)
    df.to_csv("input\preprocessed_License_Data.csv", index=False)

    # print("Successfully Saved the Prerocessed Data to input directory.")

    return df


def categorical_encode(data):
    """
    Function to encode categorical variables
    """
    try:
        cols = CATEGORICAL_FEATURES+['LICENSE_STATUS']
        final_df = pd.get_dummies(data=data, columns=cols)
    except:
        # For test data
        final_df = pd.get_dummies(data=data, columns=CATEGORICAL_FEATURES)

    return final_df


# Function to call dependent functions
def apply(data):
    print("Preprocessing started...")

    data = data_cleaning(data)
    print("Data cleanup completed...")
    print("Successfully Saved the Prerocessed Data to input directory...")

    data = categorical_encode(data)
    print("Categorical encoding completed...")

    data = data.loc[:, ~data.columns.duplicated()]

    print("Preprocessing completed...\n")
    return data