import streamlit as st
import seaborn as sns
import pandas as pd
# import datetime
import numpy as np
# import plotly.express as px
# import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
import scipy as scp

# Csss
style_css= """
<style>
div[data-testid="metric-container"] {
   background-color: rgba(28, 131, 225, 0.1);
   border: 1px solid rgba(28, 131, 225, 0.1);
   padding: 5% 5% 5% 10%;
   margins: auto;
   border-radius: 5px;
   color: rgb(30, 103, 119);
   overflow-wrap: break-word;
}
/* breakline for metric text         */
div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
   overflow-wrap: break-word;
   white-space: break-spaces;
   margins: auto;
   color: red;
}
# div[data-testid="metric-container"] > label[data-testid="stMetricValue"] > div {
# 			font-size: large;
# }

</style>

"""

def main():
    st.set_page_config(layout="wide")
    st.markdown(style_css,unsafe_allow_html=True)

    # Load xlsx data
    data_file = st.file_uploader("Upload file", type=['xlsx'])
    
    if data_file is not None:
        
        # Load Tables ************************************************************************** START 

        # Read table trip request
        if 'trip_request_tbl' not in st.session_state:
            st.session_state['trip_request_tbl'] = pd.read_excel(
                data_file,
                "trip_request",
                parse_dates={
                    "scheduled_ts_":["scheduled_ts"],
                    "pickup_ts_":["pickup_ts"],
                    "dropoff_ts_":["dropoff_ts"],
                    "requested_ts_":["requested_ts"],
                    "updated_at_":["updated_at"]},
                dtype={
                    "id":"object",
                    "created_by":"object",
                    "special_assistance":"bool",
                    "organization_id":"object",
                    "ride_type":"object",
                    "status":"object",
                    "estimated_miles":"float64",
                    "fare":"float64",
                    "discount":"float64",
                    "route_id":"object"})
            
            st.session_state['trip_request_tbl'].rename(
                columns={
                    "id":"customer_id",
                    "scheduled_ts_":"scheduled_ts", 
                    "pickup_ts_":"pickup_ts",
                    "dropoff_ts_":"dropoff_ts",
                    "requested_ts_":"requested_ts",
                    "updated_at_":"updated_at"},
                    inplace=True)
        # Trip table
        if 'trip_summary_tbl' not in st.session_state:
            st.session_state['trip_summary_tbl'] = pd.read_excel(
                data_file,
                "trip_summary",
                parse_dates={
                "start_ts_":["start_ts"],
                "end_ts_":["end_ts"]},
                dtype={
                    "route_id":"object",
                    "driver_id":"object",
                    "vehicle_id":"object",
                    "ambulatory_riders":"int8",
                    "handicapped_riders":"int8",
                    "total_riders":"int32",
                    "status":"object",
                    "trip_start_lat":"float64",
                    "trip_start_lon":"float64",
                    "trip_end_lat":"float64",
                    "trip_end_lon":"float64",
                    "min_speed_mph":"float64",
                    "max_speed_mph":"float64",
                    "avg_speed_mph":"float64",
                    "trip_distance_miles":"float64"})
            # Update date columns
            st.session_state['trip_summary_tbl'].rename(
                columns={
                    "id":"customer_id",
                    "start_ts_":"start_ts", 
                    "end_ts_":"end_ts"},
                    inplace=True)

            # Filter trip summary table to route id's observed in the trip request table
            tbl_request_routes =  st.session_state['trip_request_tbl']\
                .route_id.unique().tolist()
            st.session_state['trip_summary_tbl'] = st.session_state[
                'trip_summary_tbl'].query(f"route_id in {tbl_request_routes}")
            
            # Merge Organization Id, and Created by onto summary table.
            st.session_state['trip_summary_tbl'] = pd.merge(
                st.session_state['trip_summary_tbl'],
                st.session_state['trip_request_tbl'][
                    ["route_id","organization_id","created_by"]]\
                        .drop_duplicates(),
                        how="inner",
                        copy=False,
                        on='route_id')
            
            # Calculate trip time.
            st.session_state['trip_summary_tbl']['drive_time'] = (
                st.session_state['trip_summary_tbl']['end_ts'] - st.session_state[
                    'trip_summary_tbl']['start_ts']).astype('timedelta64[m]')
        
        # Read vehicle table
        if 'trip_vehicle_tbl' not in st.session_state:
            st.session_state['trip_vehicle_tbl'] = pd.read_excel(
                data_file,
                "vehicle",
                 dtype={
                    "id":"object",
                    "make":"object",
                    "model":"object",
                    "year":"int16",
                    "capacity":"int8",
                    "handicapped":"int8"},
                    usecols=["id","make","model","year","capacity","handicapped"])
        
        st.markdown("<br><br><br>", unsafe_allow_html=True)

        # Gather Input
        with st.container(): 
            
            col1, col2 = st.columns(2)

            # Get Date Ranges
            with col1:
                start_min_date = st.session_state['trip_summary_tbl'].start_ts.min()
                end_max_date = st.session_state['trip_summary_tbl'].end_ts.max()
                date_range = st.date_input(
                    "Enter Date Range",
                    (start_min_date, end_max_date),
                    min_value=start_min_date,
                    max_value=end_max_date)
            
          
            
            
           
            
            with col2:
                pass

            if len(date_range) == 2:
                if date_range[0] == date_range[1]:
                    metric_table = st.session_state['trip_summary_tbl']\
                        [st.session_state['trip_summary_tbl']\
                        ['start_ts'].dt.date==date_range[0]]
                else:
                    metric_table = st.session_state['trip_summary_tbl']\
                        [(st.session_state['trip_summary_tbl']\
                        ['start_ts'].dt.date>=date_range[0]) &\
                        (st.session_state['trip_summary_tbl']\
                        ['end_ts'].dt.date<=date_range[1])]
            
            
       
                # METRICS VIEW ***********************************************************************************************************
                # Row 1 Metrics
                # Note: 
                # Destinations.. 1 = 11.1 km, 2 = 1.1 km, 3 = 110 m, 4 = 11m, 5 = 1.1 m, 6 = 0.11m, 7= 11mm, 8=1.1mm
                orgs =  metric_table.organization_id.unique().tolist()
                org_options = st.multiselect(
                'Select Organizations', orgs,orgs)

                metric_table = metric_table.query("organization_id in @org_options")
                


                with st.container():
                    st.header("🗺 Trips 🧭 🚍🚐")
                    precision = 3
                    row_2_col1, row_2_col2, row_2_col3, row_2_col4 =  st.columns(4)
                    routes = metric_table.route_id.unique().tolist()
                    completed =  metric_table.query("status=='Completed'").shape[0]
                    pickup_locations = metric_table[["trip_start_lat","trip_start_lon"]].round(precision).drop_duplicates()
                    pickup_locations.rename(
                        columns={
                            "trip_start_lat":"lat",
                            "trip_start_lon":"lon"},
                            inplace=True)
                    drop_locations = metric_table[["trip_end_lat","trip_end_lon"]].round(precision).drop_duplicates()
                    drop_locations.rename(
                        columns={
                            "trip_end_lat":"lat",
                            "trip_end_lon":"lon"},
                            inplace=True)
                    
                    row_2_col1.metric(label="Routes", value=len(routes))
                    row_2_col2.metric(label="Completed Trips", value=completed)
                    row_2_col3.metric(label="Pickup Locations", value=pickup_locations.shape[0])
                    row_2_col4.metric(label="Drop off Locations", value=drop_locations.shape[0])

                    
                    mean_dis = metric_table.trip_distance_miles.mean()
                    meadian_dis = metric_table.trip_distance_miles.median()
                    max_dis = metric_table.trip_distance_miles.max()
                    min_dis = metric_table.trip_distance_miles.min()
                
                   
                    f1, f2, f3, f4 = st.columns(4)
                    f1.metric(label="Mean Trip Distance", value=f"{round(mean_dis,2)} Miles")
                    f2.metric(label="Meadian Trip Distance", value=f"{round(meadian_dis,2)} Miles")
                    f3.metric(label="Max Distance", value=f"{round(max_dis,2)} Miles")
                    f4.metric(label="Min Distance", value=f"{round(min_dis,2)} Miles")
               


                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    #************************************************************************************************************

                    d1, d2 =st.columns(2)

                    with d1:
                        
                        st.header("Pickup")
                        st.pydeck_chart(pdk.Deck(
                            map_style=None,
                            initial_view_state=pdk.ViewState(
                                latitude=39.983334,
                                longitude= -82.983330,
                                zoom=9,
                                pitch=50,
                            ),
                            layers=[
                                pdk.Layer(
                                'HexagonLayer',
                                data=pickup_locations,
                                get_position='[lon, lat]',
                                radius=200,
                                elevation_scale=2,
                                elevation_range=[0, 1000],
                                pickable=True,
                                extruded=True,
                                ),
                                pdk.Layer(
                                    'ScatterplotLayer',
                                    data=pickup_locations,
                                    get_position='[lon, lat]',
                                    get_color='[200, 30, 0, 160]',
                                    get_radius=20,
                                ),
                            ],
                        ))
                    #************************************************************************************************************
        
                    with d2:
                        st.header("Drop off")
                        st.pydeck_chart(pdk.Deck(
                            map_style=None,
                            initial_view_state=pdk.ViewState(
                                latitude=39.983334,
                                longitude= -82.983330,
                                zoom=9,
                                pitch=50,
                            ),
                            layers=[
                                pdk.Layer(
                                'HexagonLayer',
                                data=drop_locations,
                                get_position='[lon, lat]',
                                radius=200,
                                elevation_scale=2,
                                elevation_range=[0, 1000],
                                pickable=True,
                                extruded=True,
                                ),
                                pdk.Layer(
                                    'ScatterplotLayer',
                                    data=drop_locations,
                                    get_position='[lon, lat]',
                                    get_color='[200, 30, 0, 160]',
                                    get_radius=20,
                                ),
                            ],
                        ))

                st.markdown("<br><br><br>", unsafe_allow_html=True)

                with st.container():
                    st.header("🙆‍♀️🤷‍♀️🙋‍♀️🙆‍♀️ Passengers  🙆‍♀️🤷‍♀️🙋‍♀️🙆‍♀️   👩‍🦽")
                    # Row 2 Metric
                    row_3_col1, row_3_col2, row_3_col3, row_3_col4 =  st.columns(4)
                    drivers = metric_table.driver_id.unique().tolist()
                    passengers = metric_table.total_riders.values
                    Ambulatory = metric_table.ambulatory_riders.values
                    Handicapped = metric_table.handicapped_riders.values.sum()

                    row_3_col1.metric(label="Drivers", value=len(drivers))
                    row_3_col2.metric(label="Pasengers", value=passengers.sum())
                    row_3_col3.metric(label="Ambulatory", value=Ambulatory.sum())
                    row_3_col4.metric(label="Handicapped", value=Handicapped)

                    st.markdown("<br>", unsafe_allow_html=True)

    

                    riders = metric_table[['driver_id','total_riders','ambulatory_riders','handicapped_riders']]
                    riders['ambulatory_groups'] =  pd.qcut(riders['ambulatory_riders'],[0, .1, .5, 1])
                    riders.rename(columns={
                            "ambulatory_groups":"Ambulatory Groups",
                            "handicapped_riders":"Handicapped Riders"},inplace=True)

                    oc1, oc2 = st.columns(2)

                    with oc1:
                            fig_rider_amb = plt.figure(figsize=(6, 4))
                            ax = sns.histplot(
                                riders,
                                x="total_riders",
                                hue="Ambulatory Groups",
                                legend=True,)
                            ax.set(xlabel ="Passengers", ylabel = "Count", title ='Passengers:Ambulatory')
                 
                            st.pyplot(fig_rider_amb)
                    with oc2:
                            fig_rider_hand = plt.figure(figsize=(6, 4))
                            ax = sns.histplot(
                                riders,
                                x="total_riders",
                                hue="Handicapped Riders",
                                # palette="Set2",
                                legend=True,)
                            ax.set(xlabel ="Passengers", ylabel = "Count", title ='Passengers:Handicapped')
                            st.pyplot(fig_rider_hand)
                
       
                st.markdown("<br><br><br>", unsafe_allow_html=True)
                st.header("🚗🚌🚗 Vehicles 🚙🚓 🛺 🚗🚌🚗🚓 🛺 🚗")
                with st.container():
                    # Row 3 Metrics
                    row_4_col1, row_4_col2, row_4_col3, row_4_col4 =  st.columns(4)
                    vehicles = metric_table.vehicle_id.unique().tolist()
                    capacity = st.session_state['trip_vehicle_tbl'].iloc[vehicles]['capacity']
                    handicapped = st.session_state['trip_vehicle_tbl'].iloc[vehicles]['handicapped']
                    miles_traveled = metric_table.trip_distance_miles.values.tolist()

                    row_4_col1.metric(label="Vehicles", value=len(vehicles))
                    row_4_col2.metric(label="Total Capacity", value=capacity.sum())
                    row_4_col3.metric(label="handicapped Seating", value=handicapped.sum())
                    row_4_col4.metric(label="Miles traveled", value=round(sum(miles_traveled),2))
                
                vic_total_miles = metric_table[["vehicle_id","trip_distance_miles"]].groupby(['vehicle_id']).sum()
                vic_total_miles = vic_total_miles.rename_axis('vehicle_id').reset_index()
                vic_total_miles.rename(
                columns={
                    "vehicle_id":"Vehicle ID", 
                    "trip_distance_miles":"Miles"},
                    inplace=True)

                vic_average_miles = metric_table[["vehicle_id","trip_distance_miles","ambulatory_riders"]].groupby(['vehicle_id']).mean()
                vic_average_miles = vic_average_miles.rename_axis('vehicle_id').reset_index()
                vic_average_miles = vic_average_miles.sort_values("trip_distance_miles", ascending=True)
                vic_average_miles.rename(
                columns={
                    "vehicle_id":"Vehicle ID", 
                    "trip_distance_miles":"Miles"},
                    inplace=True)

                fig, ax = plt.subplots(figsize=(15,3))
                ax.hist(miles_traveled, bins=40)
                ax.set_title("Distribution Miles Traveled by all Drivers",weight="bold",fontsize=16)
                st.pyplot(fig)
                fig, ax = plt.subplots()


                oc3, oc4 = st.columns(2)
                with oc3:
                    fig_vic_tot_miles = plt.figure(figsize=(6, 4))
                    ax = sns.barplot(x="Miles", y="Vehicle ID",data=vic_total_miles, orient = 'h',
                            label="Total Miles", color="r")
                    ax.set_title("Total Miles Traveled",weight="bold")
                    st.pyplot(fig_vic_tot_miles)
                with oc4:
                    fig_vic_avg_miles = plt.figure(figsize=(6, 4))
                    ax1 = sns.barplot(x="Miles", y="Vehicle ID",data=vic_average_miles, orient = 'h',
                            label="Total Miles per Vehicle", color="b")
                    ax1.set_title("Average Miles Traveled per Vehicle",weight="bold")
                    st.pyplot(fig_vic_avg_miles)
          
                st.markdown("<br><br><br>", unsafe_allow_html=True)
                
                with st.container():
                    st.header("⌛⏲ Temporal ⌚🕰")


                    
                    metric_table['Day of the Week'] = metric_table["start_ts"].dt.day_name()
                    obs_day_mean = metric_table[['Day of the Week','trip_distance_miles']].groupby(['Day of the Week']).mean()
                    obs_day_mean = obs_day_mean.rename_axis('Day of the Week').reset_index()
                    obs_day_mean = obs_day_mean.sort_values("trip_distance_miles", ascending=True)
                    obs_day_mean.rename(
                        columns={
                            "trip_distance_miles":"Trip Distance in Miles"},
                            inplace=True)
                    
                    
                    fig_vic_tot_time= plt.figure(figsize=(20,4))
                    ax = sns.barplot(x="Day of the Week", y="Trip Distance in Miles",data=obs_day_mean)
                    ax.set_title("Mean distance traveled per Day of the Week",weight="bold",fontsize=20)
                    st.pyplot(fig_vic_tot_time)


                    obs_speed_mean = metric_table[['Day of the Week','max_speed_mph']].groupby(['Day of the Week']).mean()
                    obs_speed_mean = obs_speed_mean.rename_axis('Day of the Week').reset_index()
                    obs_speed_mean = obs_speed_mean.sort_values("max_speed_mph", ascending=True)
                    obs_speed_mean.rename(
                        columns={
                            "max_speed_mph":"Max Speed mph"},
                            inplace=True)
                    
                    fig_vic_speed_time= plt.figure(figsize=(20,4))
                    ax = sns.barplot(x="Day of the Week", y="Max Speed mph",data=obs_speed_mean)
                    ax.set_title("Mean of the Max Speed Observed per Day of the Week",weight="bold",fontsize=20)
                    st.pyplot(fig_vic_speed_time)


                    vic_tot_drive_time = metric_table[["vehicle_id","drive_time"]].groupby(['vehicle_id']).sum()
                    vic_tot_drive_time = vic_tot_drive_time.rename_axis('vehicle_id').reset_index()
                    vic_tot_drive_time = vic_tot_drive_time.sort_values("drive_time", ascending=True)
                    vic_tot_drive_time.rename(
                    columns={
                        "vehicle_id":"Vehicle ID", 
                        "drive_time":"Minutes"},
                        inplace=True)
                    
                    
                    fig_vic_tot_time= plt.figure(figsize=(20,4))
                    ax = sns.barplot(x="Minutes", y="Vehicle ID",data=vic_tot_drive_time, orient = 'h',
                                label="Total Minutes", color="y")
                    ax.set_title("Total Time Traveled by Driver",weight="bold",fontsize=20)
                    st.pyplot(fig_vic_tot_time)

                    times = metric_table[["drive_time"]].copy()
                    times_clean =  times[(np.abs(scp.stats.zscore(times)) < 3).all(axis=1)]

                
                    dis_time= plt.figure(figsize=(20,4))
                    # sns.histplot(metric_table[["drive_time"]])
                    ax2 = sns.histplot(times_clean,binwidth = 1, kde=True, legend=False)
                    ax2.set(xlabel ="Total Minutes", ylabel = "Count", title ='Distribution of Drive Times')
                    st.pyplot(dis_time)





if __name__ == '__main__':
    main()
