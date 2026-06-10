import React from 'react';

export default function ServerStatus({ isConnected }) {
  return (
    <div className="server-status" style={{ padding: '10px' }}>
      <strong>🌐 Competition Server:</strong> {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
    </div>
  );
}
