import React from 'react';

export default function VideoOverlay({ streamUrl, isLocked }) {
  return (
    <div className="video-overlay-container" style={{ border: '1px solid #333', padding: '10px' }}>
      <h3>📹 Video Overlay / Kamera Yayını</h3>
      <div style={{ position: 'relative', width: '640px', height: '480px', backgroundColor: '#000' }}>
        {isLocked && (
          <div className="lock-box" style={{ border: '2px solid red', position: 'absolute', top: '40%', left: '40%', width: '120px', height: '120px' }}>
            <span style={{ color: 'red', fontSize: '12px' }}>TARGET LOCKED</span>
          </div>
        )}
      </div>
    </div>
  );
}
