'use client';

import React, { useState } from "react";
import dynamic from "next/dynamic";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

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
    try {
      const apiBaseUrl = "http://127.0.0.1:8501"; // Ensure it matches backend
      const apiUrl = `${apiBaseUrl}/ndvi-analysis?lat=${coordinates.lat}&lng=${coordinates.lng}&year1=${years.first}&year2=${years.second}`;

      const response = await fetch(apiUrl);
      if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

      const data = await response.json();
      if (data.error) {
        alert("Error fetching NDVI data: " + data.error);
        setIsAnalyzing(false);
        return;
      }

      // Ensure the paths correctly reference the output images
      const processImageUrl = (url) =>
        url.startsWith("http") ? url : `${apiBaseUrl}/output/${url.replace("/output/", "")}`;

     
    } catch (error) {
      console.error("Error fetching NDVI data:", error);
      alert("Failed to fetch NDVI data.");
    }
    setIsAnalyzing(false);
  };

  return (
    <div className="min-h-screen bg-[#f5f9f5] p-6">
      <div className="relative h-screen w-full overflow-hidden">
        <video autoPlay loop muted className="absolute inset-0 w-full h-full object-cover brightness-50">
          <source src="/bg-video.mp4" type="video/mp4" />
          Your browser does not support the video tag.
        </video>
        <div className="absolute inset-0 flex flex-col justify-center items-center text-white">
          <h1 className="text-6xl font-bold">GreenSight</h1>
          <p className="text-lg mt-2">Analyze deforestation and afforestation changes effectively</p>
        </div>
      </div>

      <div className="mx-auto max-w-7xl space-y-8 pt-8">
        <Card>
          <CardHeader>
            <CardTitle>Input Parameters</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label>Latitude</Label>
                <Input value={coordinates.lat} onChange={(e) => setCoordinates({ ...coordinates, lat: e.target.value })} />
              </div>
              <div className="space-y-2">
                <Label>Longitude</Label>
                <Input value={coordinates.lng} onChange={(e) => setCoordinates({ ...coordinates, lng: e.target.value })} />
              </div>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label>First Year</Label>
                <Input value={years.first} onChange={(e) => setYears({ ...years, first: e.target.value })} />
              </div>
              <div className="space-y-2">
                <Label>Second Year</Label>
                <Input value={years.second} onChange={(e) => setYears({ ...years, second: e.target.value })} />
              </div>
            </div>
            <Button onClick={handleAnalyze} disabled={isAnalyzing}>{isAnalyzing ? "Analyzing..." : "Analyze Changes"}</Button>
          </CardContent>
        </Card>

        {ndviResults && (
          <>
            <div className="grid grid-cols-4 gap-6">
              <Card>
                <CardHeader><CardTitle>NDVI {years.first}</CardTitle></CardHeader>
                <CardContent><img src={ndviResults.image1} alt="NDVI Year 1" className="w-full rounded-lg" /></CardContent>
              </Card>
              <Card>
                <CardHeader><CardTitle>NDVI {years.second}</CardTitle></CardHeader>
                <CardContent><img src={ndviResults.image2} alt="NDVI Year 2" className="w-full rounded-lg" /></CardContent>
              </Card>
              <Card>
                <CardHeader><CardTitle>NDVI Change</CardTitle></CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600">Green: Afforestation, Red: Deforestation</p>
                  <img src={ndviResults.difference} alt="NDVI Change" className="w-full rounded-lg" />
                </CardContent>
              </Card>
              <Card>
                <CardHeader><CardTitle>Predicted Deforestation</CardTitle></CardHeader>
                <CardContent><img src={ndviResults.predictedDeforestation} alt="Predicted Deforestation" className="w-full rounded-lg" /></CardContent>
              </Card>
            </div>

            <Card className="mt-6">
              <CardHeader><CardTitle>Deforestation & Afforestation Summary</CardTitle></CardHeader>
              <CardContent><p>{ndviResults.summary}</p></CardContent>
            </Card>

            <Card className="mt-6">
              <CardHeader><CardTitle>NDVI Change Graph</CardTitle></CardHeader>
              <CardContent>
                <NDVIChart afforestation={ndviResults.afforestation} deforestation={ndviResults.deforestation} />
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </div>
  );
}
