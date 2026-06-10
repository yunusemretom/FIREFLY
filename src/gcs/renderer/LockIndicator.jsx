import React from 'react';

export default function LockIndicator({ isLocked }) {
  return (
    <div className="lock-indicator" style={{ padding: '10px', backgroundColor: isLocked ? '#d4edda' : '#f8d7da', color: isLocked ? '#155724' : '#721c24' }}>
      <strong>🔒 Target Lock State:</strong> {isLocked ? 'LOCKED (KİLİTLİ)' : 'SEARCHING (ARIYOR)'}
    </div>
  );
}
