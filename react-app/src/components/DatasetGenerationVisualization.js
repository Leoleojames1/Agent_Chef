import React from 'react';
import { LinearProgress, Typography, Box } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const DatasetGenerationVisualization = ({ progress }) => {
  return (
    <Box sx={{ width: '100%', mt: 4 }}>
      <Typography variant="h6" gutterBottom>Dataset Generation Progress</Typography>
      {progress.overallProgress !== undefined && (
        <>
          <LinearProgress variant="determinate" value={progress.overallProgress} sx={{ mb: 2 }} />
          <Typography variant="body2" color="text.secondary">{`${Math.round(progress.overallProgress)}% Complete`}</Typography>
        </>
      )}
      
      {progress.columnStats && (
        <Box sx={{ height: 300, mt: 4 }}>
          <Typography variant="subtitle1" gutterBottom>Column Statistics</Typography>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={Object.entries(progress.columnStats).map(([column, count]) => ({ column, count }))}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="column" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="count" fill="#8884d8" />
            </BarChart>
          </ResponsiveContainer>
        </Box>
      )}
    </Box>
  );
};

export default DatasetGenerationVisualization;