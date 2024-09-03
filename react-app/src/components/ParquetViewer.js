import React, { useState, useEffect } from 'react';
import { Typography, TableContainer, Table, TableHead, TableBody, TableRow, TableCell, Paper, TablePagination, Checkbox, Button, TextField, Box } from '@mui/material';
import axios from 'axios';

function ParquetViewer({ data, columns, totalRows, filename }) {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [pageData, setPageData] = useState([]);
  const [columnOrder, setColumnOrder] = useState(columns || []);
  const [selectedColumns, setSelectedColumns] = useState([]);
  const [editedCells, setEditedCells] = useState({});
  const [editingCell, setEditingCell] = useState(null);

  useEffect(() => {
    const fetchPageData = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/parquet_data', {
          params: {
            filename: filename,
            page: page,
            rows_per_page: rowsPerPage
          }
        });
        setPageData(response.data.content);
        setColumnOrder(response.data.columns || columns);
      } catch (error) {
        console.error('Error fetching parquet data:', error);
      }
    };

    if (filename) {
      fetchPageData();
    }
  }, [filename, page, rowsPerPage, columns]);

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
    setEditingCell(null);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
    setEditingCell(null);
  };

  const handleColumnSelect = (column) => {
    setSelectedColumns(prev => 
      prev.includes(column) ? prev.filter(c => c !== column) : [...prev, column]
    );
  };

  const handleSliceColumns = async () => {
    try {
      const response = await axios.post('http://localhost:5000/api/slice_parquet', {
        filename: filename,
        columns_to_remove: selectedColumns
      });
      alert(`${response.data.message}\nRemoved columns: ${response.data.removed_columns.join(', ')}`);
      setSelectedColumns([]);
    } catch (error) {
      console.error('Error slicing parquet:', error);
      alert('Error slicing parquet file: ' + (error.response?.data?.error || error.message));
    }
  };

  const handleCellEdit = (rowIndex, column, value) => {
    setEditedCells(prev => ({
      ...prev,
      [rowIndex]: { ...prev[rowIndex], [column]: value }
    }));
  };

  const handleSaveEdits = async () => {
    try {
      const response = await axios.post('http://localhost:5000/api/save_parquet_edits', {
        filename: filename,
        edits: editedCells
      });
      alert(response.data.message);
      setEditedCells({});
      setEditingCell(null);
      // Refresh the data after saving edits
      const fetchPageData = async () => {
        const response = await axios.get('http://localhost:5000/api/parquet_data', {
          params: {
            filename: filename,
            page: page,
            rows_per_page: rowsPerPage
          }
        });
        setPageData(response.data.content);
      };
      fetchPageData();
    } catch (error) {
      console.error('Error saving edits:', error);
      alert('Error saving edits to parquet file');
    }
  };

  const handleSaveAsNewFile = async () => {
    try {
      const response = await axios.post('http://localhost:5000/api/save_parquet_as_new', {
        filename: filename,
        edits: editedCells
      });
      alert(response.data.message);
      setEditedCells({});
      setEditingCell(null);
    } catch (error) {
      console.error('Error saving as new file:', error);
      alert('Error saving as new file');
    }
  };

  const handleCellClick = (rowIndex, column) => {
    setEditingCell({ rowIndex, column });
  };

  return (
    <>
      <Typography variant="body2" gutterBottom>
        Showing rows {page * rowsPerPage + 1} to {Math.min((page + 1) * rowsPerPage, totalRows)} out of {totalRows} total rows
      </Typography>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
        {columnOrder.map((column) => (
          <Box key={column} sx={{ display: 'flex', alignItems: 'center' }}>
            <Checkbox
              checked={selectedColumns.includes(column)}
              onChange={() => handleColumnSelect(column)}
              name={column}
            />
            <Typography variant="body2">{column}</Typography>
          </Box>
        ))}
      </Box>
      <Box sx={{ mb: 2 }}>
        <Button variant="contained" onClick={handleSliceColumns} disabled={selectedColumns.length === 0}>
          Slice Selected Columns (Save as New)
        </Button>
      </Box>
      <TableContainer component={Paper} sx={{ maxHeight: '50vh' }}>
        <Table stickyHeader aria-label="parquet data table">
          <TableHead>
            <TableRow>
              {columnOrder.map((column) => (
                <TableCell key={column}>{column}</TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {pageData.map((row, rowIndex) => (
              <TableRow key={rowIndex}>
                {columnOrder.map((column) => (
                  <TableCell key={column} onClick={() => handleCellClick(rowIndex, column)}>
                    {editingCell && editingCell.rowIndex === rowIndex && editingCell.column === column ? (
                      <TextField
                        value={editedCells[rowIndex]?.[column] ?? row[column]}
                        onChange={(e) => handleCellEdit(rowIndex, column, e.target.value)}
                        variant="standard"
                        fullWidth
                        autoFocus
                        onBlur={() => setEditingCell(null)}
                      />
                    ) : (
                      <div style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                        {editedCells[rowIndex]?.[column] ?? row[column]}
                      </div>
                    )}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      <TablePagination
        rowsPerPageOptions={[10, 25, 100, 500]}
        component="div"
        count={totalRows}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={handleChangePage}
        onRowsPerPageChange={handleChangeRowsPerPage}
      />
      <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
        <Button variant="contained" onClick={handleSaveEdits} disabled={Object.keys(editedCells).length === 0}>
          Save Edits
        </Button>
        <Button variant="contained" onClick={handleSaveAsNewFile} disabled={Object.keys(editedCells).length === 0}>
          Save As New File
        </Button>
      </Box>
    </>
  );
}

export default ParquetViewer;