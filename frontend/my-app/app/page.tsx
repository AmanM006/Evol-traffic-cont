"use client";

import dynamic from "next/dynamic";

const TrafficDashboard = dynamic(() => import("@/components/TrafficDashboard"), {
  ssr: false,
});

export default function Home() {
  return <TrafficDashboard />;
}