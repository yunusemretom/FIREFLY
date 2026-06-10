import React from 'react';

export default function TelemetryPanel({ telemetry }) {
  return (
    <div className="telemetry-panel" style={{ border: '1px solid #333', padding: '10px' }}>
      <h3>📡 Telemetry Panel / Telemetri Paneli</h3>
      <ul>
        <li>Altitude (İrtifa): {telemetry?.alt || 0} cm</li>
        <li>Speed (Hız): {telemetry?.speed || 0} cm/s</li>
        <li>Roll: {telemetry?.roll || 0}</li>
        <li>Pitch: {telemetry?.pitch || 0}</li>
        <li>Yaw: {telemetry?.yaw || 0}</li>
      </ul>
    </div>
  );
}
