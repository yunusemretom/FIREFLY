import React from 'react';

export default function StatusBar({ status }) {
  return (
    <div className="status-bar" style={{ display: 'flex', gap: '20px', padding: '10px', background: '#222', color: '#fff' }}>
      <div>System Status: <span style={{ color: '#0f0' }}>{status?.system || 'OK'}</span></div>
      <div>Battery: {status?.battery || '100'}%</div>
      <div>ARM State: {status?.armed ? 'ARMED' : 'DISARMED'}</div>
    </div>
  );
}
