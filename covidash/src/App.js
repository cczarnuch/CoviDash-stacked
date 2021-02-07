import React from 'react';
import './App.css';

import{Map, GoogleApiWrapper, Marker} from 'google-maps-react';

import CanvasJSReact from './canvasjs.react';
var CanvasJS = CanvasJSReact.CanvasJS;
var CanvasJSChart = CanvasJSReact.CanvasJSChart;

const proxyurl = 'https://cors-anywhere.herokuapp.com/';
const dataUrl = 'https://covidash.s3.ca-central-1.amazonaws.com/covidlocpreds.json';

var CovidData={'Mississauga':{'recorded':[],'prediction':[],'mean':[]},'Oakville':{'recorded':[],'prediction':[],'mean':[]},'Owen Sound':{'recorded':[],'prediction':[],'mean':[]},
'Kingston':{'recorded':[],'prediction':[],'mean':[]},'Guelph':{'recorded':[],'prediction':[],'mean':[]},'Toronto':{'recorded':[],'prediction':[],'mean':[]},
'Hamilton':{'recorded':[],'prediction':[],'mean':[]},'Thorold':{'recorded':[],'prediction':[],'mean':[]},'Waterloo':{'recorded':[],'prediction':[],'mean':[]},
'Newmarket':{'recorded':[],'prediction':[],'mean':[]},'Barrie':{'recorded':[],'prediction':[],'mean':[]},'Whitby':{'recorded':[],'prediction':[],'mean':[]},
'Port Hope':{'recorded':[],'prediction':[],'mean':[]},'Point Edward':{'recorded':[],'prediction':[],'mean':[]},'Timmins':{'recorded':[],'prediction':[],'mean':[]},
'Thunder Bay':{'recorded':[],'prediction':[],'mean':[]},'North Bay':{'recorded':[],'prediction':[],'mean':[]},'Pembroke':{'recorded':[],'prediction':[],'mean':[]},
'London':{'recorded':[],'prediction':[],'mean':[]},'Ottawa':{'recorded':[],'prediction':[],'mean':[]},'Simcoe':{'recorded':[],'prediction':[],'mean':[]},
'Windsor':{'recorded':[],'prediction':[],'mean':[]},'St. Thomas':{'recorded':[],'prediction':[],'mean':[]},'Peterborough':{'recorded':[],'prediction':[],'mean':[]},
'Kenora':{'recorded':[],'prediction':[],'mean':[]},'Stratford':{'recorded':[],'prediction':[],'mean':[]},'Brockville':{'recorded':[],'prediction':[],'mean':[]},
'Cornwall':{'recorded':[],'prediction':[],'mean':[]},'Belleville':{'recorded':[],'prediction':[],'mean':[]},'Brantford':{'recorded':[],'prediction':[],'mean':[]},
'New Liskeard':{'recorded':[],'prediction':[],'mean':[]},'Sudbury':{'recorded':[],'prediction':[],'mean':[]},'Chatham':{'recorded':[],'prediction':[],'mean':[]},
'Sault Ste. Marie':{'recorded':[],'prediction':[],'mean':[]}}

var City=[];

fetch(dataUrl)
  .then(response => response.json())
  .then(data =>{
    var count = 0;
    var counter = 34;
    var temp_values = [];
    Object.entries(data).forEach(([key, value]) => {
      for (var i=0; i< value.length; i++){
        if(value[i]['prediction']==false){
          CovidData[value[i]['loc']]['recorded'].push({y: value[i]['value'], label: key, x: count});
          if (temp_values.length>34){
            temp_values=[];
          }else{
            temp_values.push(value[i]['value']);
          }; 
        }else{
          
          CovidData[value[i]['loc']]['prediction'].push({y: value[i]['value'], label: key, x: count});
          if(counter>0){
            City.push({'loc': value[i]['loc'],'latitude': value[i]['y'], 'longitude':value[i]['x'],'cases':temp_values.shift(),'nextDayForecast': value[i]['value']})
            counter--;
          }
        }
        CovidData[value[i]['loc']]['mean'].push({y: value[i]['rolling_ave'], label: key, x: count});
      }
      count++;
    });

  })
  .catch(error => console.log('Failed because:' + {error}));



var options = {
  animationEnabled: true,	
  responsive:true,
  maintainAspectRatio: false,
  title:{
    text: "Ontario Confirmed COVID-19 Cases"
  },
  axisY : {
    title: "Confirmed Cases"
  },
  toolTip: {
    shared: true
  },
  data: [{
    type: "spline",
    name: "Prediction",
    showInLegend: true,
    dataPoints: CovidData['Toronto']['prediction']
  },
  {
    type: "spline",
    name: "Mean",
    showInLegend: true,
    dataPoints: CovidData['Toronto']['mean']
  },
  {
    type: "spline",
    name: "Recorded",
    showInLegend: true,
    dataPoints: CovidData['Toronto']['recorded']
  }
  ]
}


const mapStyles = {
  width:'100%',
  height:'100%',
  position:"fixed",
  overflow:'auto'
};

class App extends React.Component {

  constructor(props){
    super(props);
    this.state={
      cities:['Mississauga','Oakville','Owen Sound','Kingston','Guelph','Toronto','Hamilton','Thorold','Waterloo','Newmarket','Barrie','Whitby','Port Hope','Point Edward','Timmins','Thunder Bay',
              'North Bay','Pembroke','London','Ottawa','Simcoe','Windsor','St. Thomas','Peterborough','Kenora','Stratford','Brockville','Cornwall','Belleville','Brantford','New Liskeard',
              'Sudbury','Chatham','Sault Ste. Marie'],
      selectedCity: "none",
      optionselected : {
        animationEnabled: true,	
        title:{
          text: "Ontario Confirmed COVID-19 Cases"
        },
        axisY : {
          title: "Confirmed Cases"
        },
        toolTip: {
          shared: true
        },
        data: [{
          type: "spline",
          name: "Prediction",
          showInLegend: true,
          dataPoints: CovidData['Toronto']['prediction']
        },
        {
          type: "spline",
          name: "Mean",
          showInLegend: true,
          dataPoints: CovidData['Toronto']['mean']
        },
        {
          type: "spline",
          name: "Recorded",
          showInLegend: true,
          dataPoints: CovidData['Toronto']['recorded']
        }
        ]
      }
    }
  }



  selectCity(cityName){
    if (this.state.selectedCity !== "none"){
      document.getElementById(this.state.selectedCity).style.backgroundColor="#282c34";
      document.getElementById(this.state.selectedCity).style.color="white";
      document.getElementById(cityName).style.backgroundColor="white";
      document.getElementById(cityName).style.color="black";
    }else{
      document.getElementById(cityName).style.backgroundColor="white";
      document.getElementById(cityName).style.color="black";
      
    };

    this.setState({
      selectedCity: cityName,
      optionselected : {
        animationEnabled: true,	
        title:{
          text: cityName + " Confirmed COVID-19 Cases"
        },
        axisY : {
          title: "Confirmed Cases"
        },
        toolTip: {
          shared: true
        },
        data: [{
          type: "spline",
          name: "Prediction",
          showInLegend: true,
          dataPoints: CovidData[cityName]['prediction']
        },
        {
          type: "spline",
          name: "Mean",
          showInLegend: true,
          dataPoints: CovidData[cityName]['mean']
        },
        {
          type: "spline",
          name: "Recorded",
          showInLegend: true,
          dataPoints: CovidData[cityName]['recorded']
        }
        ]
      }
    })
    var cChart = document.getElementById("Cchart");
      cChart.style.visibility='visible';
  };

  setMarkerIcon(nextday, today){
    return 'https://developers.google.com/maps/documentation/javascript/examples/full/images/info-i_maps.png';
  }

  
  
  render(){
    return (
      <div className="App">
        <header className="App-header">
          <h1>Ontario Covidash </h1>
        </header>
        
        <div className="Left" >
          <header className="Ontario"> 
            <h2>Cities:</h2>

          </header>
          <nav id="nav" className="Cities">
            <button id="Mississauga" className="City" onClick={this.selectCity.bind(this,"Mississauga")}>Mississauga</button>
            <button id="Oakville" className="City" onClick={this.selectCity.bind(this,"Oakville")}>Oakville</button>
            <button id="Owen Sound" className="City" onClick={this.selectCity.bind(this,"Owen Sound")}>Owen Sound</button>
            <button id="Kingston" className="City" onClick={this.selectCity.bind(this,"Kingston")}>Kingston</button>
            <button id="Guelph" className="City" onClick={this.selectCity.bind(this,"Guelph")}>Guelph</button>
            <button id="Toronto" className="City" onClick={this.selectCity.bind(this,"Toronto")}>Toronto</button>
            <button id="Hamilton" className="City" onClick={this.selectCity.bind(this,"Hamilton")}>Hamilton</button>
            <button id="Thorold" className="City" onClick={this.selectCity.bind(this,"Thorold")}>Thorold</button>
            <button id="Waterloo" className="City" onClick={this.selectCity.bind(this,"Waterloo")}>Waterloo</button>
            <button id="Newmarket" className="City" onClick={this.selectCity.bind(this,"Newmarket")}>Newmarket</button>
            <button id="Barrie" className="City" onClick={this.selectCity.bind(this,"Barrie")}>Barrie</button>
            <button id="Whitby" className="City" onClick={this.selectCity.bind(this,"Whitby")}>Whitby</button>
            <button id="Port Hope" className="City" onClick={this.selectCity.bind(this,"Port Hope")}>Port Hope</button>
            <button id="Point Edward" className="City" onClick={this.selectCity.bind(this,"Point Edward")}>Point Edward</button>
            <button id="Timmins" className="City" onClick={this.selectCity.bind(this,"Timmins")}>Timmins</button>
            <button id="Thunder Bay" className="City" onClick={this.selectCity.bind(this,"Thunder Bay")}>Thunder Bay</button>
            <button id="North Bay" className="City" onClick={this.selectCity.bind(this,"North Bay")}>North Bay</button>
            <button id="Pembroke" className="City" onClick={this.selectCity.bind(this,"Pembroke")}>Pembroke</button>
            <button id="London" className="City" onClick={this.selectCity.bind(this,"London")}>London</button>
            <button id="Ottawa" className="City" onClick={this.selectCity.bind(this,"Ottawa")}>Ottawa</button>
            <button id="Simcoe" className="City" onClick={this.selectCity.bind(this,"Simcoe")}>Simcoe</button>
            <button id="Windsor" className="City" onClick={this.selectCity.bind(this,"Windsor")}>Windsor</button>
            <button id="St. Thomas" className="City" onClick={this.selectCity.bind(this,"St. Thomas")}>St. Thomas</button>
            <button id="Peterborough" className="City" onClick={this.selectCity.bind(this,"Peterborough")}>Peterborough</button>
            <button id="Kenora" className="City" onClick={this.selectCity.bind(this,"Kenora")}>Kenora</button>
            <button id="Stratford" className="City" onClick={this.selectCity.bind(this,"Stratford")}>Stratford</button>
            <button id="Brockville" className="City" onClick={this.selectCity.bind(this,"Brockville")}>Brockville</button>
            <button id="Cornwall" className="City" onClick={this.selectCity.bind(this,"Cornwall")}>Cornwall</button>
            <button id="Belleville" className="City" onClick={this.selectCity.bind(this,"Belleville")}>Belleville</button>
            <button id="Brantford" className="City" onClick={this.selectCity.bind(this,"Brantford")}>Brantford</button>
            <button id="New Liskeard" className="City" onClick={this.selectCity.bind(this,"New Liskeard")}>New Liskeard</button>
            <button id="Sudbury" className="City" onClick={this.selectCity.bind(this,"Sudbury")}>Sudbury</button>
            <button id="Chatham" className="City" onClick={this.selectCity.bind(this,"Chatham")}>Chatham</button>
            <button id="Marie" className="City" onClick={this.selectCity.bind(this,"Marie")}>Marie</button>
          </nav>
        </div>
        <div id='Cchart' className='Cchart'>
          <CanvasJSChart id="cityChart" options = {this.state.optionselected}/>
        </div>
        <div id='Pchart' className='Pchart'>
          <CanvasJSChart options = {options}/>
        </div>

        <div className='Map'>
        <Map
          id="map"
          google={this.props.google}
          zoom={8}
          style={mapStyles}
          initialCenter={{lat:43.5971, lng:-77.8658}}>
            
            {City.map(marker => 
            <Marker 
              key={marker.loc}
              position={{lat: marker.latitude, lng: marker.longitude}}
              icon={{
                url: this.setMarkerIcon(marker.nextDayForecast,marker.value)
              }}
              onClick={this.selectCity.bind(this,marker.loc)}
              />
          )}
          
        </Map>
          
          
        </div>
      </div>
    );
  }

}

export default GoogleApiWrapper({
  apiKey:'AIzaSyBN9PGbra-qmGwEbjqh-GM915ieTGSygFo'
})(App);

