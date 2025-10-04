'use client';

import { useState } from "react";
import dynamic from "next/dynamic";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Leaf } from "lucide-react"; // A nice icon for our input card

const NDVIChart = dynamic(() => import("./ndvi-chart"), { ssr: false });

export default function DeforestationAnalysis() {
  const [coordinates, setCoordinates] = useState({ lat: "", lng: "" });
  const [years, setYears] = useState({ first: "", second: "" });
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [ndviResults, setNDVIResults] = useState(null);

  const handleAnalyze = async () => {
    if (!coordinates.lat || !coordinates.lng || !years.first || !years.second) {
      alert("Please enter valid coordinates and select both years.");
      return;
    }

    setIsAnalyzing(true);
    setNDVIResults(null); // Clear previous results immediately for a cleaner transition
    try {
      
      const apiUrl = process.env.NEXT_PUBLIC_API_URL;
      const response = await fetch(
      `${apiUrl}/ndvi-analysis?lat=${coordinates.lat}&lng=${coordinates.lng}&year1=${years.first}&year2=${years.second}`
      );
      //const response = await fetch(
      // `http://127.0.0.1:8501/ndvi-analysis?lat=${coordinates.lat}&lng=${coordinates.lng}&year1=${years.first}&year2=${years.second}`
      //);
      if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
      const data = await response.json();
      if (data.error) {
        alert("Error fetching NDVI data: " + data.error);
        setIsAnalyzing(false);
        return;
      }

      const totalArea = parseFloat(data.afforestation_area) + parseFloat(data.deforestation_area);
      const afforestationPercent = ((parseFloat(data.afforestation_area) / totalArea) * 100).toFixed(2);
      const deforestationPercent = ((parseFloat(data.deforestation_area) / totalArea) * 100).toFixed(2);

      setNDVIResults({
        image1: data.ndvi_image1,
        image2: data.ndvi_image2,
        difference: data.ndvi_difference,
        forecast: data.predicted_deforestation,
        afforestation: parseFloat(data.afforestation_area).toFixed(2),
        deforestation: parseFloat(data.deforestation_area).toFixed(2),
        afforestationPercent,
        deforestationPercent,
        summary: `Afforestation: ${parseFloat(data.afforestation_area).toFixed(2)} sq km (${afforestationPercent}%), Deforestation: ${parseFloat(data.deforestation_area).toFixed(2)} sq km (${deforestationPercent}%)`,
      });
    } catch (error) {
      console.error("Error fetching NDVI data:", error);
      alert("Failed to fetch NDVI data.");
    }
    setIsAnalyzing(false);
  };

  return (
    <div className="min-h-screen bg-background p-4 sm:p-6 dark">
      {/* --- HERO SECTION --- */}
      <div className="relative h-[60vh] sm:h-[80vh] w-full overflow-hidden rounded-2xl shadow-2xl">
        <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent z-10" />
        <video autoPlay loop muted className="absolute inset-0 w-full h-full object-cover brightness-[.4]">
          <source src="/bg-video.mp4" type="video/mp4" />
          Your browser does not support the video tag.
        </video>
        <div className="absolute inset-0 flex flex-col justify-center items-center text-center text-white z-20 p-4">
          <h1 className="text-5xl sm:text-7xl lg:text-8xl font-black tracking-tighter text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-green-600">
            GreenSight
          </h1>
          <p className="text-md sm:text-lg max-w-2xl mt-4 text-gray-300">
            Harnessing satellite imagery and AI to monitor our planet's forests. Input your parameters below to begin.
          </p>
        </div>
      </div>

      <div className="mx-auto max-w-7xl space-y-8 py-12">
        {/* --- INPUT CARD --- */}
        <Card className="bg-black/40 backdrop-blur-xl border-2 border-white/10 shadow-lg">
          <CardHeader className="flex flex-row items-center gap-4">
            <Leaf className="text-green-500 h-8 w-8" />
            <CardTitle className="text-2xl text-green-400">Analysis Parameters</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label>Latitude</Label>
                <Input placeholder="e.g., 28.6139" className="bg-black/30" value={coordinates.lat} onChange={(e) => setCoordinates({ ...coordinates, lat: e.target.value })} />
              </div>
              <div className="space-y-2">
                <Label>Longitude</Label>
                <Input placeholder="e.g., 77.2090" className="bg-black/30" value={coordinates.lng} onChange={(e) => setCoordinates({ ...coordinates, lng: e.target.value })} />
              </div>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label>First Year</Label>
                <Input placeholder="e.g., 2015" className="bg-black/30" value={years.first} onChange={(e) => setYears({ ...years, first: e.target.value })} />
              </div>
              <div className="space-y-2">
                <Label>Second Year</Label>
                <Input placeholder="e.g., 2024" className="bg-black/30" value={years.second} onChange={(e) => setYears({ ...years, second: e.target.value })} />
              </div>
            </div>
            <Button onClick={handleAnalyze} disabled={isAnalyzing} size="lg" className="bg-green-600 hover:bg-green-500 text-white font-bold w-full sm:w-auto text-lg">
              {isAnalyzing ? "Analyzing..." : "Generate Analysis"}
            </Button>
          </CardContent>
        </Card>

        {/* --- RESULTS SECTION --- */}
        {ndviResults && (
          <div className="space-y-8 animate-fadeIn">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {[
                { title: `NDVI ${years.first}`, image: ndviResults.image1, alt: "NDVI Year 1" },
                { title: `NDVI ${years.second}`, image: ndviResults.image2, alt: "NDVI Year 2" },
                { title: "NDVI Change", image: ndviResults.difference, alt: "NDVI Change", subtitle: "Green: Afforestation, Red: Deforestation" },
                { title: "Deforestation Risk Forecast", image: ndviResults.forecast, alt: "Deforestation Forecast", subtitle: "Red areas indicate a higher probability of future deforestation." }
              ].map((item, index) => (
                <Card key={index} className="bg-black/30 backdrop-blur-xl border border-white/10 hover:border-green-500/50 hover:scale-[1.02] transition-all duration-300 overflow-hidden">
                  <CardHeader><CardTitle className="text-green-400">{item.title}</CardTitle></CardHeader>
                  <CardContent>
                    {item.subtitle && <p className="text-sm text-gray-400 mb-2">{item.subtitle}</p>}
                    <img src={item.image} alt={item.alt} className="w-full rounded-lg" />
                  </CardContent>
                </Card>
              ))}
            </div>

            <Card className="bg-black/30 backdrop-blur-xl border border-white/10">
              <CardHeader><CardTitle className="text-green-400">Analysis Summary</CardTitle></CardHeader>
              <CardContent><p className="text-lg">{ndviResults.summary}</p></CardContent>
            </Card>

            <Card className="bg-black/30 backdrop-blur-xl border border-white/10">
              <CardHeader><CardTitle className="text-green-400">Change Visualization</CardTitle></CardHeader>
              <CardContent>
                <NDVIChart afforestation={parseFloat(ndviResults.afforestation)} deforestation={parseFloat(ndviResults.deforestation)} />
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}