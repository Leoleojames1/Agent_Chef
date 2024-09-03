import React from 'react';
import { LinearProgress, Typography, Box } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

const TrainingProgressVisualization = ({ progress, metrics }) => {
  return (
    <Box sx={{ width: '100%', mt: 4 }}>
      <Typography variant="h6" gutterBottom>Training Progress</Typography>
      {progress !== null && (
        <>
          <LinearProgress variant="determinate" value={progress} sx={{ mb: 2 }} />
          <Typography variant="body2" color="text.secondary">{`${Math.round(progress)}%`}</Typography>
        </>
      )}
      
      {metrics && metrics.length > 0 && (
        <LineChart width={600} height={300} data={metrics}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="step" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="loss" stroke="#8884d8" />
          <Line type="monotone" dataKey="accuracy" stroke="#82ca9d" />
        </LineChart>
      )}
    </Box>
  );
};

export default TrainingProgressVisualization;