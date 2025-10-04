"use client"

import { Card } from "@/components/ui/card"

export default function WorldMap() {
  return (
    <Card className="overflow-hidden bg-emerald-50 p-4">
      <div className="relative aspect-[2/1] w-full">
        <img
          src="https://hebbkx1anhila5yf.public.blob.vercel-storage.com/Design%20a%20clean%20and%20modern%20webpage%20for%20a%20deforestation%20analysis%20tool.%20The%20webpage%20should%20allow%20users%20to%20input%20latitude,%20longitude,%20and%20two%20years,%20and%20display%20deforestation_afforestation%20results%20using%20NDVI%20data.%20Inclu-YALXp3SkpdQhNlQKdExRGaQf260Ukp.png"
          alt="World map showing deforestation analysis"
          className="h-full w-full object-cover opacity-50"
        />
        <div className="absolute inset-0">
          {/* Location markers */}
          <div className="absolute left-[20%] top-[30%] h-3 w-3 rounded-full bg-emerald-500 shadow-lg" />
          <div className="absolute left-[25%] top-[35%] h-3 w-3 rounded-full bg-emerald-500 shadow-lg" />
          <div className="absolute left-[60%] top-[40%] h-3 w-3 rounded-full bg-emerald-500 shadow-lg" />
          <div className="absolute left-[80%] top-[30%] h-3 w-3 rounded-full bg-emerald-500 shadow-lg" />

          {/* Location labels */}
          <div className="absolute left-[22%] top-[25%] rounded-md bg-white p-2 text-xs shadow-lg">
            Deforestation admin
          </div>
          <div className="absolute left-[45%] top-[45%] rounded-md bg-white p-2 text-xs shadow-lg">
            Coordinates admin
          </div>
          <div className="absolute right-[20%] top-[35%] rounded-md bg-white p-2 text-xs shadow-lg">Data Stream</div>
        </div>
      </div>
    </Card>
  )
}

