import ee
import geemap.foliumap as geemap
import pandas as pd 
import streamlit as st 
import json
import matplotlib.pyplot as plt
import concurrent.futures
from datetime import datetime, timedelta
import plotly.express as px
from folium.plugins import Draw
from streamlit_folium import st_folium
import statsmodels.api as sm

st.set_page_config(
page_title="SADI",
page_icon=":earth_africa:",
layout="wide"
)
st.title("STANDARDIZED AGRICULTURAL DROUGHT INDEX")
st.caption("By John Ngugi")
st.markdown("---")

#add data 

json_data = st.secrets['JSON_KEY']



json_data = json_data
# Preparing values
json_object = json.loads(json_data, strict=False)
service_account = json_object['client_email']
json_object = json.dumps(json_object)
#Authorising the app
credentials = ee.ServiceAccountCredentials(service_account, key_data=json_object)
ee.Initialize(credentials)




def maskL5(col):
    # Bits 3 and 5 are cloud shadow and cloud, respectively.
    cloudShadowBitMask = (1 << 3)
    cloudsBitMask = (1 << 5)

    # Get the pixel QA band.
    qa = col.select('QA_PIXEL')

    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0) \
    .And(qa.bitwiseAnd(cloudsBitMask).eq(0))

    return col.updateMask(mask)

coordinates = None
coordinates =[[[37.55089664907794, -1.812031669733895],
[37.55089664907794, -1.9074251114831788],
[37.84752750845294, -1.9074251114831788],
[37.84752750845294, -1.812031669733895]]]

Map = geemap.Map()

def convertCoordinates():
    new_coords = coordinates[0][0]
    x = new_coords[1]
    y= new_coords[0]
    
    return [x,y]

# Create a geemap object
Map= geemap.Map(center = convertCoordinates(), zoom=14)


# geometry = ee.Geometry.Polygon([[37.760432510614656,-2.2902863315609774],[37.98633888268497,-2.2902863315609774],[37.98633888268497,-2.1709005648732047],[37.760432510614656,-2.1709005648732047],[37.760432510614656,-2.2902863315609774]])


col1,col2,col3 = st.columns(3)

with col1:
    start_year = int(st.selectbox(
        'enter start year',
        ('2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023')
    ))


with col2:    
    end_year = int(st.selectbox(
        'enter end year',
        ('2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'),index=5)
    )


with col3:

    season = st.selectbox(
        'enter season',
        ('1','2','3','4')
    )




geometry=ee.Geometry.Polygon(coordinates)



with st.sidebar:  
    with st.form("ROI_form"):
        st.subheader("Drawn rectangular ROI parameters")
        col1,col2 = st.columns(2)
        with col1:
                xmin = st.number_input("Xmin", value=None, placeholder="Xmin")
                xmax = st.number_input("Xmax", value=None, placeholder="Xmax")
               
        with col2:    
                ymin = st.number_input("Ymin", value=None, placeholder="Ymin")
                ymax = st.number_input("Ymax", value=None, placeholder="Ymax")
                
        col1,col2 = st.columns(2)
        with col1:
                xmin_right = st.number_input("Xmin_right", value=None, placeholder="Xmin")
                xmax_right = st.number_input("Xmax_right", value=None, placeholder="Xmax")
               
        with col2:      
                ymin_right = st.number_input("Ymin_right", value=None, placeholder="Ymin")
                ymax_right = st.number_input("Ymax_right", value=None, placeholder="Ymax")
                               
        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            geometry=ee.Geometry.Polygon([[xmax,ymax],[xmax_right,ymax_right],[xmin_right,ymin_right],[xmin,ymin],])
    
    st.write('or upload a geojson file:')
    uploaded_file = st.file_uploader("Choose a geojson file",'geojson')
    if uploaded_file is not None:
        geojson_content = uploaded_file.getvalue().decode("utf-8")
        geojson = st.code(geojson_content, language="json")
        geometry=ee.Geometry(geojson)
    
 
if season=='1':
    startmonth = 1
    endmonth = 3
elif season == '2':
    startmonth = 4
    endmonth = 6
elif season =='3':
    startmonth =7
    endmonth = 9
elif season == '4':
    startmonth = 10
    endmonth = 12
else:
    print('No such season, Enter a valid season of 1-4')

landstcollection = ee.ImageCollection("LANDSAT/LE07/C02/T1") \
    .map(maskL5)\
    .filterBounds(geometry)\
    .filter(ee.Filter.calendarRange(start_year, end_year, 'year')) \
    .filter(ee.Filter.calendarRange(startmonth,endmonth, 'month'))
    



def compute_veg_indices(start_date, end_date):

        # geometry = ee.FeatureCollection("projects/ee-muthamijohn/assets/arthi-galana")
       
        # Drought_Index=Drought_Index.select('NDVI')
        # NDVI_mean = ndviL8.reduceRegion(ee.Reducer.min(), geometry, 30, maxPixels=1e9)
        # NDVI_mean = ee.Number(NDVI_mean.get('NDVI')).float().getInfo()

        def calculate_drought_index(geometry, start_date, end_date):
                # Cloud mask
            # Cloud mask
                def maskL5(col):
                        # Bits 3 and 5 are cloud shadow and cloud, respectively.
                        cloudShadowBitMask = (1 << 3)
                        cloudsBitMask = (1 << 5)

                        # Get the pixel QA band.
                        qa = col.select('QA_PIXEL')

                        # Both flags should be set to zero, indicating clear conditions.
                        mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0) \
                        .And(qa.bitwiseAnd(cloudsBitMask).eq(0))

                        return col.updateMask(mask)

                # Load the collection
                col = ee.ImageCollection("LANDSAT/LE07/C02/T1") \
                        .filterDate(start_date, end_date) \
                        .filterBounds(geometry)

                col1 = col.mean().clip(geometry)

                # Image reduction
                image = col.mean()

                # Calculate TOA spectral radiance
                ML = 0.055375
                AL = 1.18243
                TOA_radiance = image.expression('ML * B6 + AL', {
                        'ML': ML,
                        'AL': AL,
                        'B6': image.select('B6_VCID_1')
                })

                # Convert TOA spectral radiance to brightness temperature
                K1 = 607.76
                K2 = 1260.56
                brightnessTemp = TOA_radiance.expression(
                        '(K2 / (log((K1 / L) + 1))) - 273.15', {
                        'K1': K1,
                        'K2': K2,
                        'L': TOA_radiance
                        })


                clippedbrightnessTemp = brightnessTemp.clip(geometry)

                # Median
                ndvi = image.normalizedDifference(['B4', 'B3']).rename('NDVI')
                NDVI_IMAGE = ndvi.clip(geometry)

                # Find the min and max of NDVI
                min_val = ndvi.reduceRegion(ee.Reducer.min(), geometry, 30, maxPixels=1e9).get('NDVI')
                max_val = ndvi.reduceRegion(ee.Reducer.max(), geometry, 30, maxPixels=1e9).get('NDVI')
                min_value = ee.Number(min_val)
                max_value = ee.Number(max_val)

                # Fractional vegetation
                fv = ndvi.subtract(min_value).divide(max_value.subtract(min_value)).pow(2).rename('FV')
                VCI = (ndvi.subtract(min_value)).divide(max_value.subtract(min_value))

                # Emissivity
                a = ee.Number(0.004)
                b = ee.Number(0.986)
                EM = fv.multiply(a).add(b).rename('EMM')

                # Calculate land surface temperature
                landSurfaceTemp = brightnessTemp.expression(
                        '(BT / (1 + (10.60 * BT / 14388) * log(epsilon)))', {
                        'BT': brightnessTemp,
                        'epsilon': EM.select('EMM')
                        })

                # Clip the land surface temperature image to the geometry
                clippedLandSurfaceTemp = landSurfaceTemp.clip(geometry)

                # Find the min and max of LST
                min_v = clippedLandSurfaceTemp.reduceRegion(ee.Reducer.min(), geometry, 30, maxPixels=1e9).values().get(0)
                max_v = clippedLandSurfaceTemp.reduceRegion(ee.Reducer.max(), geometry, 30, maxPixels=1e9).values().get(0)
                min_LST = ee.Number(min_v)
                max_LST = ee.Number(max_v)

                max_LST_1 = ee.Image(max_LST)
                #Obtain TCI
                TCI = max_LST_1.subtract(clippedLandSurfaceTemp).divide(max_LST.subtract(min_LST))

                #Calculate VCI
                VHI = (VCI.multiply(0.5)).add(TCI.multiply(0.5))

                # VHI classification into classes based on threshold values to calculate Drought Index
                image02 = VHI.lt(0.1).And(VHI.gte(-1))
                image04 = ((VHI.gte(0.1)).And(VHI.lt(0.2))).multiply(2)
                image06 = ((VHI.gte(0.2)).And(VHI.lt(0.3))).multiply(3)
                image08 = ((VHI.gte(0.3)).And(VHI.lt(0.4))).multiply(4)
                image10 = (VHI.gte(0.4)).multiply(5)
                Drought_Index = (image02.add(image04).add(image06).add(image08).add(image10)).float()

                return Drought_Index, TCI, VCI, VHI


        Drought_Index, TCI, VCI, VHI = calculate_drought_index(geometry, start_date, end_date)

        # imageVHI =ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate(date,date).filterBounds(geometry).map(maskL8sr).median().clip(geometry)
        # VHI,VCI,TCI = vh.getVHI(imageVHI,'SR_B5','SR_B4',geometry,lst)
        VHI_mean = VHI.reduceRegion(ee.Reducer.mean(), geometry, 30, maxPixels=1e9)
        VHI_mean = ee.Number(VHI_mean.get('NDVI')).float().getInfo()
        TCI_mean = TCI.reduceRegion(ee.Reducer.mean(), geometry, 30, maxPixels=1e9)
        TCI_mean = ee.Number(TCI_mean.get('constant')).float().getInfo()
        VCI_mean = VCI.reduceRegion(ee.Reducer.mean(), geometry, 30, maxPixels=1e9)
        VCI_mean = ee.Number(VCI_mean.get('NDVI')).float().getInfo()
        Drought_Index_mean = Drought_Index.reduceRegion(ee.Reducer.mean(), geometry, 30, maxPixels=1e9)
        Drought_Index_mean = ee.Number(Drought_Index_mean.get('NDVI')).float().getInfo()


        return {'start_date': start_date, 'end_date': end_date, 'VHI_mean': VHI_mean, 'Drought_index_mean': Drought_Index_mean, 'TCI_mean': TCI_mean, 'VCI_mean': VCI_mean}

with st.spinner('Wait for it...'):
    
    dates = ee.List(landstcollection.distinct('system:time_start').aggregate_array('system:time_start')).map(
        lambda time_start: ee.Date(time_start).format('YYYY-MM-dd')).getInfo()

    # Retrieve the first sequence of dates
    first_sequence = []
    current_sequence = []

    for i, date in enumerate(dates):
        if i == 0 or int(date.split('-')[0]) >= int(dates[i-1].split('-')[0]):
            current_sequence.append(date)
        else:
            break

    first_sequence = current_sequence



    # First, convert the date strings in the first_sequence list to datetime objects
    from datetime import datetime, timedelta

    date_format = "%Y-%m-%d"
    first_sequence = [datetime.strptime(date_str, date_format) for date_str in first_sequence]

    # Create date pairs by picking each date as the start date and adding 16 days to obtain the end date
    date_pairs = []
    for i in range(len(first_sequence)):
        start_date = first_sequence[i]
        end_date = start_date + timedelta(days=16)
        date_pairs.append((start_date.strftime(date_format), end_date.strftime(date_format)))



    # Perform computations for each date pair and store the results in a list
    data = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks for each year
        futures = [executor.submit(compute_veg_indices, start_date, end_date) for start_date, end_date in date_pairs]

        # Retrieve results
        results = [future.result() for future in futures]
        data = results 
        

    # Convert the list of dictionaries to a pandas DataFrame and set 'start_date' and 'end_date' as the MultiIndex
    df= pd.DataFrame(data).set_index(['start_date', 'end_date'])





    # Step 1: Initialize an empty DataFrame to store the standardized values
    standardized_df = pd.DataFrame()

    # Step 2: Loop through each season and calculate the mean and standard deviation
    for season in df.index.get_level_values('start_date').str[:2].unique():
        seasonal_data = df.loc[df.index.get_level_values('start_date').str[:2] == season]
        min_of_season = seasonal_data['Drought_index_mean'].min()
        max_of_season = seasonal_data['Drought_index_mean'].max()
        range_of_season = max_of_season - min_of_season

        # Step 3: Compute the standardized values for the 'Drought_index_mean' column
        standardized_values = (seasonal_data['Drought_index_mean'] - min_of_season) / range_of_season * 2 - 1

        # Step 4: Assign the standardized values to the 'Standardized_Drought_Index' column
        seasonal_data['Standardized_Drought_Index'] = standardized_values

        # Append the seasonal data to the empty DataFrame
        standardized_df = pd.concat([standardized_df, seasonal_data])

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)


    #...............Mean of the standardized drought index values for each month.................#

    # Reset the MultiIndex to make 'end_date' and 'start_date' regular columns
    standardized_df.reset_index(inplace=True)

    # Drop the 'end_date' column
    standardized_df.drop('end_date', axis=1, inplace=True)

    # Convert 'start_date' to the desired format '2019-07'
    standardized_df['start_date'] = pd.to_datetime(standardized_df['start_date']).dt.strftime('%Y-%m')

    # Group by 'start_date' and compute mean for all columns
    standardized_df= standardized_df.groupby('start_date').mean().reset_index()



    #......................Mean of the indeces values for each month..............................#
    df1 = df
    # Reset the MultiIndex to make 'end_date' and 'start_date' regular columns
    df1.reset_index(inplace=True)

    # Drop the 'end_date' column
    df1.drop('end_date', axis=1, inplace=True)

    # Convert 'start_date' to the desired format '2019-07'
    df1['start_date'] = pd.to_datetime(df1['start_date']).dt.strftime('%Y-%m')

    # Group by 'start_date' and compute mean for all columns
    df1= df1.groupby('start_date').mean().reset_index()
    
    fig_standardized_drought_index = px.line(
        standardized_df,
        x="start_date",
        y = ['Standardized_Drought_Index','VHI_mean','TCI_mean','VCI_mean'],
        
       
        title="<b> Drought index mean </b>",
    )
    
    fig_standardized_drought_index.update_layout(legend = dict(
        orientation ="h",
    ))
        
    # smoothed_df = standardized_df.copy()
    # smoothed_df['Standardized_Drought_Index'] = sm.nonparametric.lowess(
    #     standardized_df['Standardized_Drought_Index'], standardized_df.index, frac=0.3
    # )[:, 1]
    # smoothed_df['VHI_mean'] = sm.nonparametric.lowess(
    #     standardized_df['VHI_mean'], standardized_df.index, frac=0.3
    # )[:, 1]
    # smoothed_df['TCI_mean'] = sm.nonparametric.lowess(
    #     standardized_df['TCI_mean'], standardized_df.index, frac=0.3
    # )[:, 1]
    # smoothed_df['VCI_mean'] = sm.nonparametric.lowess(
    #     standardized_df['VCI_mean'], standardized_df.index, frac=0.3
    # )[:, 1]

    # # Plotting the smoothed DataFrame
    # fig_standardized_drought_index = px.line(
    #     smoothed_df,
    #     x="start_date",
    #     y=['Standardized_Drought_Index', 'VHI_mean', 'TCI_mean', 'VCI_mean'],
    #     title="<b>Smoothed Drought Index Mean</b>",
    # )
    
    #....................................................#

    ######################################################

    images = []
    def calculate_drought_index(start_year, end_year):
           
            startmonth = 4
            endmonth = 6
            # Cloud mask
        # Cloud mask
            def maskL5(col):
                    # Bits 3 and 5 are cloud shadow and cloud, respectively.
                    cloudShadowBitMask = (1 << 3)
                    cloudsBitMask = (1 << 5)

                    # Get the pixel QA band.
                    qa = col.select('QA_PIXEL')

                    # Both flags should be set to zero, indicating clear conditions.
                    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0) \
                    .And(qa.bitwiseAnd(cloudsBitMask).eq(0))

                    return col.updateMask(mask)

            # Load the collection
            col = ee.ImageCollection("LANDSAT/LE07/C02/T1") \
                    .map(maskL5) \
                    .filter(ee.Filter.calendarRange(start_year, end_year, 'year')) \
                    .filter(ee.Filter.calendarRange(startmonth,endmonth, 'month'))\
                    .filterBounds(geometry)

            col1 = col.mean().clip(geometry)

            # Image reduction
            image = col.mean()

            # Calculate TOA spectral radiance
            ML = 0.055375
            AL = 1.18243
            TOA_radiance = image.expression('ML * B6 + AL', {
                    'ML': ML,
                    'AL': AL,
                    'B6': image.select('B6_VCID_1')
            })

            # Convert TOA spectral radiance to brightness temperature
            K1 = 607.76
            K2 = 1260.56
            brightnessTemp = TOA_radiance.expression(
                    '(K2 / (log((K1 / L) + 1))) - 273.15', {
                    'K1': K1,
                    'K2': K2,
                    'L': TOA_radiance
                    })


            clippedbrightnessTemp = brightnessTemp.clip(geometry)

            # Median
            ndvi = image.normalizedDifference(['B4', 'B3']).rename('NDVI')
            NDVI_IMAGE = ndvi.clip(geometry)

            # Find the min and max of NDVI
            min_val = ndvi.reduceRegion(ee.Reducer.min(), geometry, 30, maxPixels=1e9).get('NDVI')
            max_val = ndvi.reduceRegion(ee.Reducer.max(), geometry, 30, maxPixels=1e9).get('NDVI')
            min_value = ee.Number(min_val)
            max_value = ee.Number(max_val)

            # Fractional vegetation
            fv = ndvi.subtract(min_value).divide(max_value.subtract(min_value)).pow(2).rename('FV')
            VCI = (ndvi.subtract(min_value)).divide(max_value.subtract(min_value))

            # Emissivity
            a = ee.Number(0.004)
            b = ee.Number(0.986)
            EM = fv.multiply(a).add(b).rename('EMM')

            # Calculate land surface temperature
            landSurfaceTemp = brightnessTemp.expression(
                    '(BT / (1 + (10.60 * BT / 14388) * log(epsilon)))', {
                    'BT': brightnessTemp,
                    'epsilon': EM.select('EMM')
                    })

            # Clip the land surface temperature image to the geometry
            clippedLandSurfaceTemp = landSurfaceTemp.clip(geometry)

            # Find the min and max of LST
            min_v = clippedLandSurfaceTemp.reduceRegion(ee.Reducer.min(), geometry, 30, maxPixels=1e9).values().get(0)
            max_v = clippedLandSurfaceTemp.reduceRegion(ee.Reducer.max(), geometry, 30, maxPixels=1e9).values().get(0)
            min_LST = ee.Number(min_v)
            max_LST = ee.Number(max_v)

            max_LST_1 = ee.Image(max_LST)
            #Obtain TCI
            TCI = max_LST_1.subtract(clippedLandSurfaceTemp).divide(max_LST.subtract(min_LST))

            #Calculate VCI
            VHI = (VCI.multiply(0.5)).add(TCI.multiply(0.5))

            # VHI classification into classes based on threshold values to calculate Drought Index
            image02 = VHI.lt(0.1).And(VHI.gte(-1))
            image04 = ((VHI.gte(0.1)).And(VHI.lt(0.2))).multiply(2)
            image06 = ((VHI.gte(0.2)).And(VHI.lt(0.3))).multiply(3)
            image08 = ((VHI.gte(0.3)).And(VHI.lt(0.4))).multiply(4)
            image10 = (VHI.gte(0.4)).multiply(5)
            Drought_Index = (image02.add(image04).add(image06).add(image08).add(image10))

            Drought_index_sd_image = Drought_Index.expression('(2*((Drought_Index - min_DI)/(max_DI - min_DI)) -1)',{
                'Drought_Index': Drought_Index,
                'min_DI' : ee.Number(Drought_Index.reduceRegion(ee.Reducer.min(), geometry, 30, maxPixels=1e9).values().get(0)),
                'max_DI' :ee.Number(Drought_Index.reduceRegion(ee.Reducer.max(), geometry, 30, maxPixels=1e9).values().get(0)),

            })



            images.append(Drought_index_sd_image)
            images.append(Drought_Index)
            min_value_DI = Drought_Index.reduceRegion(ee.Reducer.min(), geometry, 30, maxPixels=1e9).values().get(0)
            max_value_DI = Drought_Index.reduceRegion(ee.Reducer.max(), geometry, 30, maxPixels=1e9).values().get(0)
            print(max_value_DI.getInfo())
            print(min_value_DI.getInfo())
            min_value = Drought_index_sd_image.reduceRegion(ee.Reducer.min(), geometry, 30, maxPixels=1e9).values().get(0)
            max_value = Drought_index_sd_image.reduceRegion(ee.Reducer.max(), geometry, 30, maxPixels=1e9).values().get(0)
            print(max_value.getInfo())
            print(min_value.getInfo())
            return Drought_index_sd_image
        
    calculate_drought_index(start_year, end_year)    


    Map.add_basemap('HYBRID')
    Map.addLayer(images[0],{'min':-1,'max':1,'palette':['#ec0000','#ecca00','#ec9b00','#ec5300','#ec2400']},'Drought index standardized image')
   
    url = images[0].getDownloadUrl({
                    'name':'image',
                    'scale': 30,
                    'crs': 'EPSG:4326',
                    'region': geometry,
                    'format':"GEO_TIFF"
                })
    

    # Convert the map to a Streamlit-compatible format
    
    Map.to_streamlit(height=400) 
    # create columns for the graph and dataframe
    st.dataframe(standardized_df,use_container_width=True)
    st.plotly_chart(fig_standardized_drought_index,use_container_width=True)
    
with st.sidebar:  
    st.write("Download The CSV data here:")    
    st.write('filters')
    TCI_checkbox= st.checkbox('TCI')
    VCI_checkbox = st.checkbox('VCI')
    VHI_checkbox = st.checkbox('VHI')
    Drought_index_checkbox = st.checkbox('Drought Index')
    all_checkbox= st.checkbox('All')
    
    download_df = pd.DataFrame()
    # Add selected columns to download_df based on checkboxes
    if TCI_checkbox:
        download_df['TCI'] = standardized_df['TCI_mean']

    if VCI_checkbox:
        download_df['VCI'] =standardized_df['VCI_mean']

    if VHI_checkbox:
        download_df['VHI'] = standardized_df['VHI_mean']

    if Drought_index_checkbox:
        download_df['Drought_Index'] = standardized_df['Standardized_Drought_Index']
    
    if all_checkbox:
        # Create a copy of the original DataFrame
        download_df = standardized_df.copy()
    
    
    st.caption('Preview')
    # Display the resulting DataFrame
    st.write(download_df)

    
    st.download_button('Download CSV',download_df.to_csv(),'data.csv', 'text/csv')
    st.markdown('#####')
    st.write("Download map Image here:")      
    st.markdown("Link to image: [link](%s)" % url)
    


# TODO 
# get the image Download url (done)
# get the filters for csv data 
# add input for geometry (done)
# if possible get to understand geemap draw features 