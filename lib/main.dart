import 'package:flutter/material.dart';
import 'package:here_sdk/core.dart';
import 'package:here_sdk/core.engine.dart';
import 'package:here_sdk/core.errors.dart';
import 'package:here_sdk/mapview.dart';

void main() {
  SdkContext.init(IsolateOrigin.main);
  _initializeHERESDK();
  runApp(const MyApp());
}
void _initializeHERESDK() async {
  // Needs to be called before accessing SDKOptions to load necessary libraries.
  SdkContext.init(IsolateOrigin.main);

  // Set your credentials for the HERE SDK.
  String accessKeyId = "FowwLtNmV2I6dpENRk2zAg";
  String accessKeySecret = "EJTfj-5gYLyhv73w2LWG-Di3_liYI6CEfeJH1UQYlZoluT9WEOZyU0Z5KzNTpFTC49qo-QyZnJTeEYGz30e7Ng";
  SDKOptions sdkOptions = SDKOptions.withAccessKeySecret(accessKeyId, accessKeySecret);

  try {
    await SDKNativeEngine.makeSharedInstance(sdkOptions);
  } on InstantiationException {
    throw Exception("Failed to initialize the HERE SDK.");
  }
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  MapPolygon _createMapCircle() {
  double radiusInMeters = 300;
  GeoCircle geoCircle = GeoCircle(GeoCoordinates(28.449281, 77.583474), radiusInMeters);

  GeoPolygon geoPolygon = GeoPolygon.withGeoCircle(geoCircle);
  Color fillColor = Color.fromARGB(160, 0, 144, 138);
  MapPolygon mapPolygon = MapPolygon(geoPolygon, fillColor);

  return mapPolygon;
}

  // This widget is the root of your application.
  @override
  static const primaryColor = Color.fromARGB(255, 0, 0, 0);
  
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        // This is the theme of your application.
        //
        // Try running your application with "flutter run". You'll see the
        // application has a blue toolbar. Then, without quitting the app, try
        // changing the primarySwatch below to Colors.green and then invoke
        // "hot reload" (press "r" in the console where you ran "flutter run",
        // or simply save your changes to "hot reload" in a Flutter IDE).
        // Notice that the counter didn't reset back to zero; the application
        // is not restarted.
        primarySwatch: Colors.blue,
        primaryColor: primaryColor
      ),
      home: Scaffold(
        appBar: AppBar(
          centerTitle: true,
          title: Text(
          'Next Location'),
          backgroundColor: primaryColor,
          leading: Icon(Icons.map),
        ) ,
        
        body: HereMap(
          onMapCreated: _onMapCreated, createMapCircle: _createMapCircle,
        ),
        
        )
    );
  }
  void _onMapCreated(HereMapController hereMapController){
    hereMapController.mapScene
      .loadSceneForMapScheme(MapScheme.normalNight, (error) {
        if (error!=null){
          print('Error'+ error.toString());
          return;
        }
        double distencetoEarthInMeters=8000;
        hereMapController.camera.lookAtPointWithDistance(GeoCoordinates(28.449281, 77.583474), distencetoEarthInMeters);
       });
      hereMapController.mapScene.enableFeatures(
          {MapFeatures.trafficFlow: MapFeatureModes.trafficFlowWithFreeFlow});
      hereMapController.mapScene.enableFeatures(
          {MapFeatures.trafficIncidents: MapFeatureModes.defaultMode});
      hereMapController.mapScene.enableFeatures(
          {MapFeatures.extrudedBuildings: MapFeatureModes.defaultMode});
      hereMapController.mapScene.enableFeatures(
          {MapFeatures.buildingFootprints: MapFeatureModes.defaultMode});

      MapPolygon pp = _createMapCircle();
      hereMapController.mapScene.addMapPolygon(pp);
  }
  
}