import React from 'react';

export default function MapView({ coordinates }) {
  return (
    <div className="map-view-container" style={{ border: '1px solid #333', padding: '10px' }}>
      <h3>🗺️ Map View / Harita Görünümü</h3>
      <p>Latitude: {coordinates?.lat || "N/A"}</p>
      <p>Longitude: {coordinates?.lon || "N/A"}</p>
    </div>
  );
}
