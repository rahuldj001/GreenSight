"use client"

import { MapPin } from "lucide-react"

export default function LocationMap() {
  return (
    <div className="relative h-full w-full overflow-hidden rounded-lg bg-emerald-50">
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="text-center">
          <MapPin className="mx-auto h-8 w-8 text-emerald-600" />
          <p className="mt-2 text-sm text-emerald-700">Click on the map to select location</p>
        </div>
      </div>
    </div>
  )
}

