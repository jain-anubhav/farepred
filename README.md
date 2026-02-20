# farepred
Predict prices

data description

flt_id: flight id
pathod_id: market id
airline: airline name
dfd: days before departure
dephour: flight departure hour	
depdow: day of week of departure date
depdt: flight departure date
target_price: price of the flight
expected_minfare: expected minimum price


I have a dataset with flight and prices. Each flight is indicated by departure date (depdt), operating carrier (airline), market (pathod_id), flight id (flt_id). dephour indicates the departure hour of the flight. Flights in a departure date and market are competing with each other. dfd is the time dimension indicating days before departure. 0 dfd is the last day before the flight departs. target_price is the price for the flight on a dfd. expected_minfare is the expected lowest price for a dfd and is always known for any dfd. 

On a certain dfd x for a flight, prices for future dfds (dfd<x) are not known for a flight and we would like to predict them using a model. Prices for earlier dfds (dfd<=x) are known and can be used as features for future dfd prediction. I have a base model prediction for each flight as of dfd x for future dfd y (y < x) as max(price on dfd x, mincpf on dfd y). 

Objective: Test and iterate on ML models to help do a better prediction than base model on a given dfd 40? Models can be trained using train_data and tested on test_data.
