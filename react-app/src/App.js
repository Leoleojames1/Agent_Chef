import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import axios from 'axios';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Drawer from '@mui/material/Drawer';
import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemText from '@mui/material/ListItemText';
import Divider from '@mui/material/Divider';
import Checkbox from '@mui/material/Checkbox';
import FormControlLabel from '@mui/material/FormControlLabel';
import FormGroup from '@mui/material/FormGroup';
import Switch from '@mui/material/Switch';
import { styled } from '@mui/system';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import TablePagination from '@mui/material/TablePagination';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogTitle from '@mui/material/DialogTitle';
import { TreeView, TreeItem } from '@mui/x-tree-view';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import Collapse from '@mui/material/Collapse';
import ListSubheader from '@material-ui/core/ListSubheader';
import { FormControl, InputLabel } from '@mui/material';

// Your custom components
import CustomPromptEditor from './components/CustomPromptEditor';
import TrainingProgressVisualization from './components/TrainingProgressVisualization';
import ParquetViewer from './components/ParquetViewer';

// AceEditor imports
import AceEditor from 'react-ace';
import 'ace-builds/src-noconflict/mode-json';
import 'ace-builds/src-noconflict/mode-python';
import 'ace-builds/src-noconflict/mode-javascript';
import 'ace-builds/src-noconflict/mode-html';
import 'ace-builds/src-noconflict/mode-css';
import 'ace-builds/src-noconflict/mode-text';
import 'ace-builds/src-noconflict/theme-monokai';

const drawerWidth = 240;

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#4fc3f7',
    },
    secondary: {
      main: '#f48fb1',
    },
    background: {
      default: 'transparent',
      paper: 'rgba(48, 48, 48, 0.8)',
    },
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backdropFilter: 'blur(10px)',
          backgroundColor: 'rgba(18, 18, 18, 0.8)',
        },
      },
    },
  },
});

const PerlinNoiseBackground = styled('div')({
  position: 'fixed',
  top: 0,
  left: 0,
  width: '100%',
  height: '100%',
  zIndex: -1,
  background: `
    linear-gradient(to bottom right, 
      rgba(0, 0, 50, 0.8), 
      rgba(0, 0, 0, 0.8)
    ),
    url("data:image/svg+xml,%3Csvg viewBox='0 0 400 400' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.15' numOctaves='3' stitchTiles='stitch'/%3E%3Canimate attributeName='baseFrequency' values='0.15; 0.2; 0.25; 0.2' dur='40s' repeatCount='indefinite'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E")
  `,
  backgroundBlendMode: 'overlay',
  animation: 'noise 40s steps(4) infinite',
});

const MainContent = styled(Box)({
  flexGrow: 1,
  padding: theme.spacing(3),
  transition: theme.transitions.create('margin', {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.leavingScreen,
  }),
});

const ActionButton = styled(Button)(({ theme }) => ({
  margin: theme.spacing(1),
}));

function App() {
  const [mode, setMode] = useState('');
  const [ollamaModel, setOllamaModel] = useState('');
  const [ingredientFiles, setIngredientFiles] = useState([]);
  const [dishFiles, setDishFiles] = useState([]);
  const [latexFiles, setLatexFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [editorContent, setEditorContent] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('');
  const [output, setOutput] = useState('');
  const [filename, setFilename] = useState('');
  const [templates, setTemplates] = useState({});
  const [error, setError] = useState(null);
  const [selectedTemplate, setSelectedTemplate] = useState('');
  const [syntheticTechnique, setSyntheticTechnique] = useState('');
  const [newTemplateName, setNewTemplateName] = useState('');
  const [newTemplateColumns, setNewTemplateColumns] = useState('');
  const [numSamples, setNumSamples] = useState(100);
  const [newTemplateFields, setNewTemplateFields] = useState('');
  const [seeds, setSeeds] = useState([]);
  const [selectedSeed, setSelectedSeed] = useState(null);
  const [seedMessage, setSeedMessage] = useState('No seeds available');  // Add this line
  const [parquetData, setParquetData] = useState(null);
  const [parquetColumns, setParquetColumns] = useState([]);
  const [totalRows, setTotalRows] = useState(0);
  const [fileType, setFileType] = useState(null);  // Add this line
  const [ollamaModels, setOllamaModels] = useState([]);
  const [selectedOllamaModel, setSelectedOllamaModel] = useState('');
  const [customSeedFilename, setCustomSeedFilename] = useState('');
  const [customSeedText, setCustomSeedText] = useState('');
  const [txtContent, setTxtContent] = useState('');
  const [txtFilename, setTxtFilename] = useState('');
  const [proceduralParquet, setProceduralParquet] = useState(null);
  const [selectedLatexFile, setSelectedLatexFile] = useState('');
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [editorMode, setEditorMode] = useState('text');
  const [builtSeed, setBuiltSeed] = useState(null);
  const [allFiles, setAllFiles] = useState([]);
  const [selectedFileType, setSelectedFileType] = useState(null);
  const [formattingMode, setFormattingMode] = useState('manual');
  const aceEditorRef = useRef(null);
  const [sampleRate, setSampleRate] = useState(100); // Default to 100%
  const [totalSamples, setTotalSamples] = useState(0);
  const [selectedSample, setSelectedSample] = useState(null);
  const [numParaphrases, setNumParaphrases] = useState(5);
  const [paraphrases, setParaphrases] = useState([]);
  const [paraphrasesPerSample, setParaphrasesPerSample] = useState(5);
  const [staticColumns, setStaticColumns] = useState([]);
  const [availableColumns, setAvailableColumns] = useState([]);
  const [columnTypes, setColumnTypes] = useState({});
  const [useAllSamples, setUseAllSamples] = useState(true);
  const [customPrompts, setCustomPrompts] = useState({});
  const [unslothModelName, setUnslothModelName] = useState('');
  const [unslothTrainingEpochs, setUnslothTrainingEpochs] = useState(1);
  const [unslothBatchSize, setUnslothBatchSize] = useState(2);
  const [unslothAccumulationSteps, setUnslothAccumulationSteps] = useState(4);
  const [openCustomTemplateDialog, setOpenCustomTemplateDialog] = useState(false);
  const [customTemplateName, setCustomTemplateName] = useState('');
  const [customTemplateContent, setCustomTemplateContent] = useState('');
  const [customTemplates, setCustomTemplates] = useState({});
  const [huggingfaceFolders, setHuggingfaceFolders] = useState([]);
  const [selectedTrainingFile, setSelectedTrainingFile] = useState('');
  const [selectedValidationFile, setSelectedValidationFile] = useState('');
  const [selectedHuggingfaceModel, setSelectedHuggingfaceModel] = useState('');
  const [newModelName, setNewModelName] = useState('');
  const [customChatTemplate, setCustomChatTemplate] = useState('');
  const [selectedTestFile, setSelectedTestFile] = useState('');
  const [showTrainingProgress, setShowTrainingProgress] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(null);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [pageData, setPageData] = useState([]);
  const [columnOrder, setColumnOrder] = useState([]);
  const [validationSplit, setValidationSplit] = useState(0);
  const [precision, setPrecision] = useState('4bit');
  const [mergeBaseModel, setMergeBaseModel] = useState('');
  const [mergeAdapterModel, setMergeAdapterModel] = useState('');
  const [mergeOutputName, setMergeOutputName] = useState('');
  const [unslothMode, setUnslothMode] = useState('train');
  const [adapterFiles, setAdapterFiles] = useState([]);
  const [dequantizeOption, setDequantizeOption] = useState('no');
  const [ggufInputModel, setGgufInputModel] = useState('');
  const [ggufOutputName, setGgufOutputName] = useState('');
  const [ggufOuttype, setGgufOuttype] = useState('f16');
  const [ovenModels, setOvenModels] = useState([]);
  const [dequantizeInputModel, setDequantizeInputModel] = useState('');
  const [dequantizeOutputName, setDequantizeOutputName] = useState('');
  const [dequantizePrecision, setDequantizePrecision] = useState('f16');
  const [ggufQuantizeInputModel, setGgufQuantizeInputModel] = useState('');
  const [ggufQuantizeOutputName, setGgufQuantizeOutputName] = useState('');
  const [ggufQuantizeType, setGgufQuantizeType] = useState('');
  const [selectedGgufInfo, setSelectedGgufInfo] = useState(null);
  const [ggufFiles, setGgufFiles] = useState([]);
  const [expandedSections, setExpandedSections] = useState({
    ingredients: true,
    dishes: true,
    salads: true,
    edits: true,
    oven: true,
    latex: true
  });
  
  const toggleSection = (section) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }));
  };

  useEffect(() => {
    fetchAdapterFiles();
  }, []);
  
  useEffect(() => {
    if (mergeAdapterModel) {
      setMergeOutputName(mergeAdapterModel.replace('checkpoint-', ''));
    }
  }, [mergeAdapterModel]);
  
  const fetchAdapterFiles = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/adapter-files');
      setAdapterFiles(response.data.adapter_files);
      setOvenModels(response.data.oven_models); // Add this line
    } catch (error) {
      console.error('Error fetching adapter and oven model files:', error);
      setError('Error fetching adapter and oven model files');
    }
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/templates');
        console.log("Fetched templates:", response.data);
        setTemplates(response.data || {});
      } catch (error) {
        console.error('Error fetching templates:', error);
        setError('Error fetching templates: ' + error.message);
      }
    };

    fetchData();
  }, []);

  useEffect(() => {
    fetchFiles();
    fetchTemplates();
    fetchSeeds();
    fetchOllamaModels();
    fetchGgufFiles();
  }, []);

useEffect(() => {
  if (availableColumns.length > 0) {
    const initialColumnTypes = availableColumns.reduce((acc, column) => {
      if (column === 'command_description') acc[column] = 'static';
      else if (column === 'command') acc[column] = 'reference';
      else acc[column] = 'dynamic';
      return acc;
    }, {});
    setColumnTypes(initialColumnTypes);
    console.log("Initialized column types:", initialColumnTypes);
  }
}, [availableColumns]);

useEffect(() => {
  if (selectedTrainingFile && selectedHuggingfaceModel) {
    const suggestedName = `${selectedHuggingfaceModel}_${selectedTrainingFile.split('.')[0]}`;
    setNewModelName(suggestedName);
  }
}, [selectedTrainingFile, selectedHuggingfaceModel]);

useEffect(() => {
  const updatedCustomPrompts = { ...customPrompts };
  Object.entries(columnTypes).forEach(([column, type]) => {
    if (type === 'dynamic' && !updatedCustomPrompts.dynamicColumns?.[column]) {
      updatedCustomPrompts.dynamicColumns = {
        ...updatedCustomPrompts.dynamicColumns,
        [column]: {
          system: `You are an AI assistant specializing in generating ${column} data.`,
          user: `Given the context and reference values, generate a suitable ${column} response:`
        }
      };
    }
  });
  setCustomPrompts(updatedCustomPrompts);
}, [columnTypes]);

  if (error) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Box sx={{ p: 3 }}>
          <Typography color="error">{error}</Typography>
        </Box>
      </ThemeProvider>
    );
  }

  const handleGgufFileSelection = async (file) => {
    try {
      const response = await axios.get(`http://localhost:5000/api/gguf-info/${file}`);
      setSelectedGgufInfo(response.data);
    } catch (error) {
      console.error('Error fetching GGUF file info:', error);
      setError('Error fetching GGUF file info');
    }
  };
  
  // Add this new function to fetch GGUF files and their info
  const fetchGgufFiles = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/gguf-files');
      setGgufFiles(response.data.gguf_files);
    } catch (error) {
      console.error('Error fetching GGUF files:', error);
      setError('Error fetching GGUF files');
    }
  };

  const handleColumnTypeToggle = (column) => {
    setColumnTypes(prevTypes => ({
      ...prevTypes,
      [column]: prevTypes[column] === 'static' ? 'dynamic' : 'static'
    }));
  };
  
  const handleColumnTypeChange = (column, newType) => {
    setColumnTypes(prevTypes => ({
      ...prevTypes,
      [column]: newType
    }));
  };

  const fetchOllamaModels = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/ollama-models');
      setOllamaModels(response.data.models);
    } catch (error) {
      console.error('Error fetching Ollama models:', error);
      setError('Error fetching Ollama models');
    }
  };

  const handleFileSelection = (file) => {
    setSelectedFiles(prevSelected => {
      if (prevSelected.some(f => f.name === file.name && f.type === file.type)) {
        return prevSelected.filter(f => f.name !== file.name || f.type !== file.type);
      } else {
        return [...prevSelected, file];
      }
    });
  };

  const saveTxtFile = async () => {
    try {
      const response = await axios.post('http://localhost:5000/api/save', {
        filename: txtFilename,
        content: txtContent,
        type: 'ingredient',
      });
      if (response.data.success) {
        alert('Text file saved successfully');
        fetchFiles();
      } else {
        setError(`Error saving text file: ${response.data.message}`);
      }
    } catch (error) {
      setError(`Error saving text file: ${error.response?.data?.message || error.message}`);
    }
  };

  const buildProceduralParquet = async () => {
    try {
      const response = await axios.post('http://localhost:5000/api/convert_to_parquet', {
        content: txtContent,
        template: selectedTemplate,
        filename: txtFilename,
      });
      setProceduralParquet(response.data.parquet_file);
      setOutput(`Procedural parquet created: ${response.data.parquet_file}`);
      fetchFiles();
    } catch (error) {
      setError(error.response?.data?.error || error.message);
    }
  };

  const generateSyntheticParquet = async () => {
    try {
      if (!proceduralParquet) {
        throw new Error("No procedural parquet file selected.");
      }
      
      const response = await axios.post('http://localhost:5000/api/generate_synthetic', {
        seed_parquet: proceduralParquet,
        num_samples: numSamples,
        ollama_model: selectedOllamaModel,
        system_prompt: systemPrompt,
      });
      
      setOutput(`Synthetic parquet created: ${response.data.filename}`);
      fetchFiles();
    } catch (error) {
      setError(error.response?.data?.error || error.message);
    }
  };

  const useLatexFile = async (filename) => {
    try {
      const response = await axios.get(`http://localhost:5000/api/file/latex/${filename}`);
      setTxtContent(response.data.content);
      setTxtFilename(filename.replace('.tex', '.txt'));
      setSelectedLatexFile(filename);
    } catch (error) {
      setError(`Error loading LaTeX file: ${error.response?.data?.message || error.message}`);
    }
  };

  const fetchSeeds = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/seeds');
      if (response.data.seeds && response.data.seeds.length > 0) {
        setSeeds(response.data.seeds);
        setSeedMessage('');  // Clear the message if seeds are available
      } else {
        setSeeds([]);
        setSeedMessage('No seeds available');
      }
    } catch (error) {
      console.error('Error fetching seeds:', error);
      setError('Error fetching seeds');
      setSeedMessage('Error fetching seeds');
    }
  };

  const buildSeed = async () => {
    try {
      const response = await axios.post('http://localhost:5000/api/convert_to_json_parquet', {
        filename,
      });
      setBuiltSeed(response.data.parquet_file);
      setOutput(`Seed parquet created: ${response.data.parquet_file}`);
      fetchFiles();
    } catch (error) {
      setError(error.response?.data?.error || error.message);
    }
  };

  const combineSelectedFiles = async () => {
    if (selectedFiles.length < 2) {
      setError("Please select at least two files to combine.");
      return;
    }
    
    const fileTypes = new Set(selectedFiles.map(file => file.name.split('.').pop()));
    if (fileTypes.size > 1) {
      setError("Can only combine files of the same type.");
      return;
    }

    try {
      const response = await axios.post('http://localhost:5000/api/combine_files', {
        files: selectedFiles,
      });
      setOutput(`Combined file created: ${response.data.combined_file}`);
      fetchFiles();
      setSelectedFiles([]);  // Clear selection after combining
    } catch (error) {
      setError(error.response?.data?.error || error.message);
    }
  };

  const fetchFiles = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/files');
      setIngredientFiles(response.data.ingredient_files);
      setDishFiles(response.data.dish_files);
      setLatexFiles(response.data.latex_files);
      setHuggingfaceFolders(response.data.huggingface_folders);
      
      const allFiles = [
        ...response.data.ingredient_files.map(file => ({ name: file, type: 'ingredient' })),
        ...response.data.dish_files.map(file => ({ name: file, type: 'dish' })),
        ...response.data.latex_files.map(file => ({ name: file, type: 'latex' })),
        ...response.data.salad_files.map(file => ({ name: file, type: 'salad' })),
        ...response.data.oven_files.map(file => ({ name: file, type: 'oven' })),
        ...response.data.edits_files.map(file => ({ name: file, type: 'edits' })),
      ];
  
      setAllFiles(allFiles);
    } catch (error) {
      console.error('Error fetching files:', error);
      setError('Error fetching files');
    }
  };

  const renderFileSection = (title, files, type) => {
    const mainFiles = files.filter(file => !file.name.startsWith('edits/'));
    const editFiles = files.filter(file => file.name.startsWith('edits/'));
  
    return (
      <>
        <ListItem>
          <ListItemButton onClick={() => toggleSection(type)}>
            {expandedSections[type] ? <ExpandMoreIcon /> : <ChevronRightIcon />}
            <Typography variant="h6">{title}</Typography>
          </ListItemButton>
        </ListItem>
        <Collapse in={expandedSections[type]} timeout="auto" unmountOnExit>
          <List component="div" disablePadding>
            {mainFiles.map(file => (
              <ListItem 
                key={`${file.type}-${file.name}`} 
                disablePadding 
                sx={{
                  backgroundColor: selectedFiles.some(f => f.name === file.name && f.type === file.type) 
                    ? 'rgba(25, 118, 210, 0.12)' 
                    : 'transparent',
                  pl: 4,
                }}
              >
                <ListItemButton 
                  onClick={() => handleFileSelection(file)}
                  onDoubleClick={() => selectFile(file.name, file.type)}
                >
                  <ListItemText primary={file.name} />
                </ListItemButton>
              </ListItem>
            ))}
            {editFiles.length > 0 && (
              <>
                <ListItem>
                  <ListItemButton onClick={() => toggleSection(`${type}-edits`)}>
                    {expandedSections[`${type}-edits`] ? <ExpandMoreIcon /> : <ChevronRightIcon />}
                    <Typography variant="subtitle1">Edits</Typography>
                  </ListItemButton>
                </ListItem>
                <Collapse in={expandedSections[`${type}-edits`]} timeout="auto" unmountOnExit>
                  <List component="div" disablePadding>
                    {editFiles.map(file => (
                      <ListItem 
                        key={`${file.type}-${file.name}`} 
                        disablePadding 
                        sx={{
                          backgroundColor: selectedFiles.some(f => f.name === file.name && f.type === file.type) 
                            ? 'rgba(25, 118, 210, 0.12)' 
                            : 'transparent',
                          pl: 6,
                        }}
                      >
                        <ListItemButton 
                          onClick={() => handleFileSelection(file)}
                          onDoubleClick={() => selectFile(file.name.replace('edits/', ''), file.type)}
                        >
                          <ListItemText primary={file.name.replace('edits/', '')} />
                        </ListItemButton>
                      </ListItem>
                    ))}
                  </List>
                </Collapse>
              </>
            )}
          </List>
        </Collapse>
      </>
    );
  };

  const selectSample = async (filename) => {
    try {
      const response = await axios.get(`http://localhost:5000/api/file/ingredient/${filename}`);
      setSelectedSample(response.data.content[0]);  // Assuming the first row is selected
    } catch (error) {
      console.error('Error fetching sample content:', error);
      setError('Error fetching sample content');
    }
  };

  const selectSeed = async (filename) => {
    try {
      const response = await axios.get(`http://localhost:5000/api/file/ingredient/${filename}`);
      setSelectedSeed(filename);
      setEditorContent(JSON.stringify(response.data.content, null, 2));
    } catch (error) {
      console.error('Error fetching seed content:', error);
      setError('Error fetching seed content');
    }
  };

  const fetchTemplates = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/templates');
      console.log("Fetched templates:", response.data);
      setTemplates(response.data || {});
    } catch (error) {
      console.error('Error fetching templates:', error);
      setError('Error fetching templates');
    }
  };

  const selectFile = async (filename, type) => {
    setSelectedFile(filename);
    setSelectedFileType(type);
    setFilename(filename);
    
    const isEditFile = type === 'edits' || filename.startsWith('edits/');
    const actualFilename = isEditFile ? filename.replace('edits/', '') : filename;
  
    try {
      if (actualFilename.endsWith('.parquet')) {
        const response = await axios.get(`http://localhost:5000/api/parquet_data`, {
          params: {
            filename: isEditFile ? `edits/${actualFilename}` : filename,
            page: 0,
            rows_per_page: 100
          }
        });
        setFileType('parquet');
        setParquetData({
          content: response.data.content,
          filename: actualFilename
        });
        setParquetColumns(response.data.columns);
        setTotalRows(response.data.total_rows);
        setAvailableColumns(response.data.columns);
        setEditorContent('');
  
        // Initialize column types
        const initialColumnTypes = response.data.columns.reduce((acc, column) => {
          if (column === 'task' || column === 'instruction') {
            acc[column] = 'static';
          } else if (column === 'command') {
            acc[column] = 'reference';
          } else {
            acc[column] = 'dynamic';
          }
          return acc;
        }, {});
        setColumnTypes(initialColumnTypes);
      } else {
        const apiEndpoint = isEditFile 
          ? `http://localhost:5000/api/file/edit/${actualFilename}`
          : `http://localhost:5000/api/file/${type}/${filename}`;
        
        const response = await axios.get(apiEndpoint);
        setFileType(actualFilename.split('.').pop());
        if (typeof response.data.content === 'string') {
          setEditorContent(response.data.content);
        } else {
          setEditorContent(JSON.stringify(response.data.content, null, 2));
        }
        setParquetData(null);
        setParquetColumns([]);
        setTotalRows(0);
      }
      
      // Set editor mode based on file extension
      if (actualFilename.endsWith('.json')) {
        setEditorMode('json');
      } else if (actualFilename.endsWith('.py')) {
        setEditorMode('python');
      } else if (actualFilename.endsWith('.js')) {
        setEditorMode('javascript');
      } else if (actualFilename.endsWith('.html')) {
        setEditorMode('html');
      } else if (actualFilename.endsWith('.css')) {
        setEditorMode('css');
      } else {
        setEditorMode('text');
      }
    } catch (error) {
      console.error('Error fetching file content:', error);
      setError(`Error fetching file content: ${error.response?.data?.error || error.message}`);
      setParquetData(null);
      setParquetColumns([]);
      setTotalRows(0);
      setEditorContent('');
    }
  };

  const handleOpenCustomTemplateDialog = () => {
    setOpenCustomTemplateDialog(true);
  };
  
  const handleCloseCustomTemplateDialog = () => {
    setOpenCustomTemplateDialog(false);
    setCustomTemplateName('');
    setCustomTemplateContent('');
  };

  const handleColumnToggle = (column) => {
    setStaticColumns(prev => 
      prev.includes(column) 
        ? prev.filter(c => c !== column) 
        : [...prev, column]
    );
  };

  const handleSaveCustomTemplate = () => {
    if (customTemplateName && customTemplateContent) {
      setCustomTemplates(prev => ({
        ...prev,
        [customTemplateName]: customTemplateContent
      }));
      handleCloseCustomTemplateDialog();
    }
  };

  const insertSymbol = (symbol) => {
    const editor = aceEditorRef.current.editor;
    editor.insert(`$("${symbol}")`);
    editor.focus();
  };

  const parseDataset = async () => {
    try {
      const response = await axios.post('http://localhost:5000/api/parse_dataset', {
        content: txtContent,
        template: selectedTemplate,
        mode: formattingMode,
      });
      if (response.data.success) {
        setOutput(JSON.stringify(response.data.result, null, 2));
        fetchFiles();
      } else {
        setError(`Error parsing dataset: ${response.data.message}`);
      }
    } catch (error) {
      setError(`Error parsing dataset: ${error.response?.data?.message || error.message}`);
    }
  };
  
  const saveFile = async () => {
    try {
      const response = await axios.post('http://localhost:5000/api/save', {
        filename,
        content: editorContent,
        type: 'ingredient',
      }, {
        headers: {
          'Content-Type': 'application/json'
        }
      });
      if (response.data.success) {
        alert('File saved successfully');
        fetchFiles();
      } else {
        setError(`Error saving file: ${response.data.message}`);
      }
    } catch (error) {
      console.error('Error saving file:', error.response?.data || error.message);
      setError(`Error saving file: ${error.response?.data?.message || error.message}`);
    }
  };

  const convertToJson = async () => {
    try {
      if (!selectedFile || !selectedTemplate) {
        throw new Error("Please select a file and a template");
      }
  
      let content = editorContent;
      if (fileType === 'txt') {
        // For TXT files, we need to send the content as is
        const response = await axios.get(`http://localhost:5000/api/file/${selectedFileType}/${selectedFile}`);
        content = response.data.content;
      }
  
      const response = await axios.post('http://localhost:5000/api/convert_to_json', {
        filename: selectedFile,
        content: content,
        template: selectedTemplate,
        fileType: selectedFileType
      });
  
      setOutput(`JSON file created: ${response.data.json_file}, Parquet file created: ${response.data.parquet_file}`);
      fetchFiles();
    } catch (error) {
      setError(error.response?.data?.error || error.message);
    }
  };

  const convertJsonToParquetSeed = async () => {
    try {
      if (!selectedFile || !selectedFile.endsWith('.json')) {
        throw new Error("Please select a JSON file to convert to parquet");
      }

      const response = await axios.post('http://localhost:5000/api/convert_to_json_parquet', {
        filename: selectedFile,
      });

      setBuiltSeed(response.data.parquet_file);
      setOutput(`Seed parquet created: ${response.data.parquet_file}`);
      fetchFiles();
    } catch (error) {
      setError(error.response?.data?.error || error.message);
    }
  };

  const convertToJsonParquet = async () => {
    try {
      const response = await axios.post('http://localhost:5000/api/convert_to_json_parquet', {
        content: editorContent,
        template: selectedTemplate,
        filename,
      });
      setOutput(`JSON file: ${response.data.json_file}, Parquet file: ${response.data.parquet_file}`);
      fetchFiles();
    } catch (error) {
      setError(error.response?.data?.error || error.message);
    }
  };

  const saveCustomSeed = async () => {
    try {
      const response = await axios.post('http://localhost:5000/api/save', {
        filename: customSeedFilename,
        content: customSeedText,
        type: 'ingredient',
      });
      if (response.data.success) {
        alert('Custom seed file saved successfully');
        fetchFiles();
        fetchSeeds();
      } else {
        setError(`Error saving custom seed file: ${response.data.message}`);
      }
    } catch (error) {
      console.error('Error saving custom seed file:', error.response?.data || error.message);
      setError(`Error saving custom seed file: ${error.response?.data?.message || error.message}`);
    }
  };

  const useCustomSeedForSynthetic = async () => {
    try {
      const response = await axios.post('http://localhost:5000/api/convert_to_json_parquet', {
        content: customSeedText,
        template: selectedTemplate,
        filename: customSeedFilename,
      });
      setSelectedSeed(response.data.parquet_file);
      setOutput(`Seed parquet created: ${response.data.parquet_file}`);
      fetchFiles();
      fetchSeeds();
    } catch (error) {
      setError(error.response?.data?.error || error.message);
    }
  };
  
  const generateParaphrases = async () => {
    if (!selectedSample) {
      setError("Please select a sample first");
      return;
    }
  
    try {
      const response = await axios.post('http://localhost:5000/api/generate_paraphrases', {
        sample: selectedSample,
        num_paraphrases: numParaphrases,
        ollama_model: selectedOllamaModel
      });
      
      setParaphrases(response.data.paraphrases);
      setOutput(`Generated ${response.data.paraphrases.length} paraphrases`);
    } catch (error) {
      setError(error.response?.data?.error || error.message);
    }
  };

  const handleSavePrompts = (prompts) => {
    setCustomPrompts(prompts);
  };

  const runAgentChef = async () => {
    try {
      setError(null);
      setOutput("Processing...");
      
      if (!selectedFile) {
        throw new Error("Please select a file for generating synthetic data.");
      }
      
      if (!selectedOllamaModel) {
        throw new Error("No Ollama model selected. Please select a model.");
      }
      
      const dataToSend = {
        mode: 'custom',
        ollamaModel: selectedOllamaModel,
        seedFile: selectedFile,
        systemPrompt: customPrompts.system,
        sampleRate: sampleRate,
        paraphrasesPerSample: paraphrasesPerSample,
        columnTypes: columnTypes,
        useAllSamples: useAllSamples,
        customPrompts: customPrompts,
        customChatTemplate: customTemplates[selectedTemplate] || null
      };
  
      console.log("Sending data to server:", JSON.stringify(dataToSend, null, 2));
  
      const response = await axios.post('http://localhost:5000/api/run', dataToSend);
      
      if (response.data.error) {
        setError(response.data.error);
        setOutput('');
      } else {
        setOutput(JSON.stringify(response.data, null, 2));
        fetchFiles();
      }
    } catch (error) {
      setError(error.response?.data?.error || error.message);
      setOutput('');
    }
  };

  const addTemplate = async () => {
    try {
      await axios.post('http://localhost:5000/api/templates', {
        name: newTemplateName,
        fields: newTemplateFields.split(',').map(field => field.trim())
      });
      fetchTemplates();
      setNewTemplateName('');
      setNewTemplateFields('');
    } catch (error) {
      console.error('Error adding template:', error);
      setError(`Error adding template: ${error.response?.data?.message || error.message}`);
    }
  };

  const renderTemplateOptions = () => {
    if (!templates || Object.keys(templates).length === 0) {
      return <MenuItem value="">No templates available</MenuItem>;
    }
    return Object.entries(templates).map(([name, fields]) => (
      <MenuItem key={name} value={name}>
        {name}: {Array.isArray(fields) ? fields.join(', ') : 'Invalid template'}
      </MenuItem>
    ));
  };

  const runDequantization = async () => {
    try {
      setError(null);
      setOutput("Processing dequantization...");
      
      if (!dequantizeInputModel) {
        throw new Error("Please select a model to dequantize.");
      }
      
      const dataToSend = {
        inputPath: dequantizeInputModel,
        outputName: dequantizeOutputName || dequantizeInputModel,
        precision: dequantizePrecision
      };
  
      console.log("Sending data for dequantization:", JSON.stringify(dataToSend, null, 2));
  
      const response = await axios.post('http://localhost:5000/api/dequantize', dataToSend);
    
      if (response.data.error) {
        setError(response.data.error);
        setOutput('');
      } else {
        setOutput(JSON.stringify(response.data, null, 2));
        fetchFiles();
      }
    } catch (error) {
      setError(error.response?.data?.error || error.message);
      setOutput('');
    }
  };

  const runUnslothMerge = async () => {
    try {
      setError(null);
      setOutput("Processing Unsloth merge...");
      
      if (!mergeBaseModel || !mergeAdapterModel || !mergeOutputName) {
        throw new Error("Please select a base model, an adapter model, and provide an output name.");
      }
      
      const dataToSend = {
        baseModelPath: mergeBaseModel,
        adapterPath: mergeAdapterModel,
        outputPath: mergeOutputName,
        dequantize: dequantizeOption
      };
  
      console.log("Sending data to Unsloth merger:", JSON.stringify(dataToSend, null, 2));
  
      const response = await axios.post('http://localhost:5000/api/merge_adapter', dataToSend);
    
      if (response.data.error) {
        setError(response.data.error);
        setOutput('');
      } else {
        setOutput(JSON.stringify(response.data, null, 2));
        fetchFiles();
      }
    } catch (error) {
      setError(error.response?.data?.error || error.message);
      setOutput('');
    }
  };
  

  // Modify the existing runGgufConversion function
  const runGgufConversion = async () => {
    try {
      setError(null);
      setOutput("Processing GGUF conversion...");
      
      if (!ggufInputModel) {
        throw new Error("Please select a model to convert to GGUF.");
      }
      
      const dataToSend = {
        inputPath: ggufInputModel,
        outputName: ggufOutputName || ggufInputModel,
        quantizationType: ggufOuttype
      };

      console.log("Sending data for GGUF conversion:", JSON.stringify(dataToSend, null, 2));

      const response = await axios.post('http://localhost:5000/api/convert_to_gguf', dataToSend);
    
      if (response.data.error) {
        setError(response.data.error);
        setOutput('');
      } else {
        setOutput(JSON.stringify(response.data, null, 2));
        fetchGgufFiles();
      }
    } catch (error) {
      setError(error.response?.data?.error || error.message);
      setOutput('');
    }
  };

  const runGgufQuantization = async () => {
    try {
      setError(null);
      setOutput("Processing GGUF quantization...");
      
      if (!ggufQuantizeInputModel) {
        throw new Error("Please select a GGUF model to quantize.");
      }
      
      const dataToSend = {
        inputPath: ggufQuantizeInputModel,
        outputName: ggufQuantizeOutputName || ggufQuantizeInputModel,
        bits: parseInt(ggufQuantizeType.substring(1))  // Convert 'q4_0' to 4, 'q8_0' to 8, etc.
      };
  
      console.log("Sending data for GGUF quantization:", JSON.stringify(dataToSend, null, 2));
  
      const response = await axios.post('http://localhost:5000/api/quantize_gguf', dataToSend);
    
      if (response.data.error) {
        setError(response.data.error);
        setOutput('');
      } else {
        setOutput(JSON.stringify(response.data, null, 2));
        fetchGgufFiles();
      }
    } catch (error) {
      setError(error.response?.data?.error || error.message);
      setOutput('');
    }
  };

  const runUnslothTraining = async () => {
    try {
      setError(null);
      setOutput("Processing Unsloth training...");
      
      if (!selectedTrainingFile || !selectedHuggingfaceModel) {
        throw new Error("Please select a training file and a Hugging Face model.");
      }
      
      const dataToSend = {
        trainingFile: selectedTrainingFile,
        validationFile: selectedValidationFile,
        testFile: selectedTestFile,
        huggingfaceModel: selectedHuggingfaceModel,
        newModelName: newModelName,
        numTrainEpochs: unslothTrainingEpochs,
        perDeviceTrainBatchSize: unslothBatchSize,
        gradientAccumulationSteps: unslothAccumulationSteps,
        precision: precision
      };
  
      console.log("Sending data to Unsloth trainer:", JSON.stringify(dataToSend, null, 2));
  
      const response = await axios.post('http://localhost:5000/api/unsloth_train', dataToSend);
    
      if (response.data.error) {
        setError(response.data.error);
        setOutput('');
      } else {
        setOutput(JSON.stringify(response.data, null, 2));
        fetchFiles();
      }
    } catch (error) {
      setError(error.response?.data?.error || error.message);
      setOutput('');
    }
  };

        return (
          <ThemeProvider theme={theme}>
            <CssBaseline />
            <PerlinNoiseBackground />
            <Box sx={{ display: 'flex' }}>
              <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
                <Toolbar>
                  <Typography variant="h4" noWrap component="div">
                    🍲Agent Chef🥩
                  </Typography>
                </Toolbar>
              </AppBar>
              <Drawer
                variant="permanent"
                sx={{
                  width: drawerWidth,
                  flexShrink: 0,
                  [`& .MuiDrawer-paper`]: { width: drawerWidth, boxSizing: 'border-box' },
                }}
              >
                <Toolbar />
                <Box sx={{ overflow: 'auto' }}>
                  <List>
                    {renderFileSection("Ingredients", allFiles.filter(file => file.type === 'ingredient'), 'ingredients')}
                    <Divider />
                    {renderFileSection("Dishes", allFiles.filter(file => file.type === 'dish'), 'dishes')}
                    <Divider />
                    {renderFileSection("Salads", allFiles.filter(file => file.type === 'salad'), 'salads')}
                    <Divider />
                    {renderFileSection("Edits", allFiles.filter(file => file.type === 'edits'), 'edits')}
                    <Divider />
                    <ListItem>
                      <ListItemButton onClick={() => toggleSection('oven')}>
                        {expandedSections['oven'] ? <ExpandMoreIcon /> : <ChevronRightIcon />}
                        <Typography variant="h6">Oven</Typography>
                      </ListItemButton>
                    </ListItem>
                    <Collapse in={expandedSections['oven']} timeout="auto" unmountOnExit>
                      <List component="div" disablePadding>
                        {huggingfaceFolders.map((folder, index) => (
                          <ListItem key={`oven-${index}`} disablePadding sx={{ pl: 4 }}>
                            <ListItemButton>
                              <ListItemText primary={folder} />
                            </ListItemButton>
                          </ListItem>
                        ))}
                      </List>
                    </Collapse>
                    {latexFiles.length > 0 && (
                      <>
                        <Divider />
                        {renderFileSection("LaTeX Files", allFiles.filter(file => file.type === 'latex'), 'latex')}
                      </>
                    )}
                  </List>
                </Box>
              </Drawer>
              <MainContent>
                <Toolbar />
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
                      <Typography variant="h5" gutterBottom>Setup</Typography>
                      <Select
                        fullWidth
                        value={selectedOllamaModel}
                        onChange={(e) => setSelectedOllamaModel(e.target.value)}
                        displayEmpty
                        sx={{ mb: 2 }}
                      >
                        <MenuItem value="">Select Ollama Model</MenuItem>
                        {ollamaModels.map((model) => (
                          <MenuItem key={model} value={model}>{model}</MenuItem>
                        ))}
                      </Select>
                      <Select
                        fullWidth
                        value={selectedTemplate}
                        onChange={(e) => setSelectedTemplate(e.target.value)}
                        displayEmpty
                        sx={{ mb: 2 }}
                      >
                        <MenuItem value="">Select Template</MenuItem>
                        {renderTemplateOptions()}
                      </Select>
                      <TextField
                        fullWidth
                        label="Random Sample Rate (%)"
                        type="number"
                        value={sampleRate}
                        onChange={(e) => setSampleRate(Math.max(0, Math.min(100, parseInt(e.target.value) || 0)))}
                        inputProps={{ min: 0, max: 100 }}
                        sx={{ mb: 2 }}
                      />
                      <TextField
                        fullWidth
                        label="Paraphrases per Sample"
                        type="number"
                        value={paraphrasesPerSample}
                        onChange={(e) => setParaphrasesPerSample(Math.max(1, parseInt(e.target.value) || 1))}
                        inputProps={{ min: 1 }}
                        sx={{ mb: 2 }}
                      />
                      <FormControlLabel
                        control={
                          <Switch
                            checked={useAllSamples}
                            onChange={(e) => setUseAllSamples(e.target.checked)}
                          />
                        }
                        label="Use All Samples (Negates Random Sample Rate)"
                      />
                    </Paper>
                  </Grid>
                  <Grid item xs={12}>
                    <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
                      <Typography variant="h5" gutterBottom>Editor / Viewer</Typography>
                      <Typography variant="body2" sx={{ mb: 2 }}>
                        Note: You can include multi-line content, such as Python code, within a single $("data/code") group split across multiple lines. Run the Parse Dataset button to have Agent Chef attempt to generate your $("data") formatting for the provided txt file and based on the selected format or format the data by hand.
                      </Typography>
                      <Box sx={{ mb: 2 }}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={formattingMode === 'automatic'}
                              onChange={(e) => setFormattingMode(e.target.checked ? 'automatic' : 'manual')}
                            />
                          }
                          label="Automatic Formatting"
                        />
                      </Box>
                      <TextField
                        fullWidth
                        label="Filename"
                        value={filename}
                        onChange={(e) => setFilename(e.target.value)}
                        sx={{ mb: 2 }}
                      />
                      {fileType === 'parquet' ? (
                        <ParquetViewer 
                          data={parquetData} 
                          columns={parquetColumns} 
                          totalRows={totalRows} 
                          filename={selectedFile}
                        />
                      ) : (
                        <AceEditor
                          ref={aceEditorRef}
                          mode={editorMode}
                          theme="monokai"
                          onChange={setEditorContent}
                          value={editorContent}
                          name="editor"
                          editorProps={{ $blockScrolling: true }}
                          width="100%"
                          height="400px"
                          setOptions={{
                            useWorker: false,
                            showLineNumbers: true,
                            tabSize: 2,
                          }}
                        />
                      )}
                      <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between' }}>
                        <Button variant="contained" onClick={saveFile}>
                          Save as TXT
                        </Button>
                        <Button 
                          variant="contained" 
                          onClick={convertToJson}
                          disabled={!selectedTemplate || !selectedFile || !(fileType === 'txt' || fileType === 'tex' || fileType === 'json')}
                        >
                          Convert to JSON & Parquet Seeds
                        </Button>
                        <Button 
                          variant="contained" 
                          onClick={combineSelectedFiles}
                          disabled={selectedFiles.length < 2}
                        >
                          Combine Selected Files
                        </Button>
                        <Button variant="contained" onClick={parseDataset}>
                          Parse Dataset
                        </Button>
                      </Box>
                      {selectedFiles.length > 0 && (
                        <Box sx={{ mt: 2 }}>
                          <Typography variant="subtitle1">Selected Files:</Typography>
                          <ul>
                            {selectedFiles.map(file => (
                              <li key={`${file.type}-${file.name}`}>{file.name} ({file.type})</li>
                            ))}
                          </ul>
                        </Box>
                      )}
                    </Paper>
                  </Grid>
                  <Grid item xs={12}>
                    <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
                      <Typography variant="h5" gutterBottom>Column Selection</Typography>
                      <Typography variant="body2" gutterBottom>
                        Select the type for each column:
                        - Static: Copied directly from the original
                        - Reference: Used for augmenting dynamic columns
                        - Dynamic: Paraphrased/generated
                      </Typography>
                      {availableColumns.map((column) => (
                        <Box key={column} sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                          <Typography sx={{ minWidth: 150 }}>{column}:</Typography>
                          <Select
                            value={columnTypes[column] || 'dynamic'}
                            onChange={(e) => handleColumnTypeChange(column, e.target.value)}
                            sx={{ minWidth: 150 }}
                          >
                            <MenuItem value="static">Static</MenuItem>
                            <MenuItem value="reference">Reference</MenuItem>
                            <MenuItem value="dynamic">Dynamic</MenuItem>
                          </Select>
                        </Box>
                      ))}
                    </Paper>
                  </Grid>
                  <Grid item xs={12}>
                    <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
                      <CustomPromptEditor
                        columnTypes={columnTypes}
                        onSavePrompts={handleSavePrompts}
                        initialCustomPrompts={customPrompts}
                      />
                    </Paper>
                  </Grid>
                  <Grid item xs={12}>
                    <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
                      <Typography variant="h5" gutterBottom>Synthetic Data Generation</Typography>
                      <Button 
                        fullWidth
                        variant="contained" 
                        onClick={runAgentChef}
                        disabled={!selectedOllamaModel || !selectedFile || !(fileType === 'parquet' || fileType === 'txt' || fileType === 'json')}
                        sx={{ mb: 2 }}
                      >
                        Generate Synthetic Data
                      </Button>
                    </Paper>
                  </Grid>
                  <Grid item xs={12}>
                    <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
                      <Typography variant="h5" gutterBottom>Unsloth Training and Merging</Typography>
                      <FormControl fullWidth sx={{ mb: 2 }}>
                        <InputLabel>Operation Mode</InputLabel>
                        <Select
                          value={unslothMode}
                          onChange={(e) => setUnslothMode(e.target.value)}
                        >
                          <MenuItem value="train">Train</MenuItem>
                          <MenuItem value="merge">Merge</MenuItem>
                          <MenuItem value="gguf_convert">GGUF Convert</MenuItem>
                          <MenuItem value="dequantize">Dequantize</MenuItem>
                          <MenuItem value="quantize">Quantize GGUF</MenuItem>
                        </Select>
                      </FormControl>

                      {unslothMode === 'train' && (
                        <>
                          <Typography variant="h6" gutterBottom>Training Configuration</Typography>
                          <Select
                            fullWidth
                            value={selectedTrainingFile}
                            onChange={(e) => setSelectedTrainingFile(e.target.value)}
                            displayEmpty
                            sx={{ mb: 2 }}
                          >
                            <MenuItem value="">Select Training File</MenuItem>
                            {allFiles
                              .filter(file => file.name.endsWith('.parquet'))
                              .map((file) => (
                                <MenuItem key={file.name} value={file.name}>{file.name}</MenuItem>
                              ))}
                          </Select>
                          <Select
                            fullWidth
                            value={selectedValidationFile}
                            onChange={(e) => setSelectedValidationFile(e.target.value)}
                            displayEmpty
                            sx={{ mb: 2 }}
                          >
                            <MenuItem value="">Select Validation File (Optional)</MenuItem>
                            {allFiles
                              .filter(file => file.name.endsWith('.parquet'))
                              .map((file) => (
                                <MenuItem key={file.name} value={file.name}>{file.name}</MenuItem>
                              ))}
                          </Select>
                          <Select
                            fullWidth
                            value={selectedTestFile}
                            onChange={(e) => setSelectedTestFile(e.target.value)}
                            displayEmpty
                            sx={{ mb: 2 }}
                          >
                            <MenuItem value="">Select Test File (Optional)</MenuItem>
                            {allFiles
                              .filter(file => file.name.endsWith('.parquet'))
                              .map((file) => (
                                <MenuItem key={file.name} value={file.name}>{file.name}</MenuItem>
                              ))}
                          </Select>
                          <Select
                            fullWidth
                            value={selectedHuggingfaceModel}
                            onChange={(e) => setSelectedHuggingfaceModel(e.target.value)}
                            displayEmpty
                            sx={{ mb: 2 }}
                          >
                            <MenuItem value="">Select Hugging Face Model</MenuItem>
                            {huggingfaceFolders.map((folder) => (
                              <MenuItem key={folder} value={folder}>{folder}</MenuItem>
                            ))}
                          </Select>
                          <TextField
                            fullWidth
                            label="New Model Name"
                            value={newModelName}
                            onChange={(e) => setNewModelName(e.target.value)}
                            sx={{ mb: 2 }}
                          />
                          <Select
                            fullWidth
                            value={precision}
                            onChange={(e) => setPrecision(e.target.value)}
                            sx={{ mb: 2 }}
                          >
                            <MenuItem value="4bit">4-bit Precision</MenuItem>
                            <MenuItem value="16bit">16-bit Precision</MenuItem>
                          </Select>
                          <TextField
                            fullWidth
                            label="Number of Training Epochs"
                            type="number"
                            value={unslothTrainingEpochs}
                            onChange={(e) => setUnslothTrainingEpochs(Math.max(1, parseInt(e.target.value) || 1))}
                            inputProps={{ min: 1 }}
                            sx={{ mb: 2 }}
                          />
                          <TextField
                            fullWidth
                            label="Batch Size"
                            type="number"
                            value={unslothBatchSize}
                            onChange={(e) => setUnslothBatchSize(Math.max(1, parseInt(e.target.value) || 1))}
                            inputProps={{ min: 1 }}
                            sx={{ mb: 2 }}
                          />
                          <TextField
                            fullWidth
                            label="Gradient Accumulation Steps"
                            type="number"
                            value={unslothAccumulationSteps}
                            onChange={(e) => setUnslothAccumulationSteps(Math.max(1, parseInt(e.target.value) || 1))}
                            inputProps={{ min: 1 }}
                            sx={{ mb: 2 }}
                          />
                        </>
                      )}

                      {unslothMode === 'merge' && (
                        <>
                          <Typography variant="h6" gutterBottom>Merge Configuration</Typography>
                          <Select
                            fullWidth
                            value={mergeBaseModel}
                            onChange={(e) => setMergeBaseModel(e.target.value)}
                            displayEmpty
                            sx={{ mb: 2 }}
                          >
                            <MenuItem value="">Select Base Model (from Hugging Face folder)</MenuItem>
                            {huggingfaceFolders.map((folder) => (
                              <MenuItem key={folder} value={folder}>{folder}</MenuItem>
                            ))}
                          </Select>
                          <Select
                            fullWidth
                            value={mergeAdapterModel}
                            onChange={(e) => setMergeAdapterModel(e.target.value)}
                            displayEmpty
                            sx={{ mb: 2 }}
                          >
                            <MenuItem value="">Select Adapter Model (from Oven folder)</MenuItem>
                            {adapterFiles.map((file) => (
                              <MenuItem key={file} value={file}>{file}</MenuItem>
                            ))}
                          </Select>
                          <TextField
                            fullWidth
                            label="Output Model Name"
                            value={mergeOutputName}
                            onChange={(e) => setMergeOutputName(e.target.value)}
                            sx={{ mb: 2 }}
                          />
                          <FormControl fullWidth sx={{ mb: 2 }}>
                            <InputLabel>Dequantize LoRA</InputLabel>
                            <Select
                              value={dequantizeOption}
                              onChange={(e) => setDequantizeOption(e.target.value)}
                            >
                              <MenuItem value="no">No dequantization</MenuItem>
                              <MenuItem value="f16">Dequantize to Float16</MenuItem>
                              <MenuItem value="f32">Dequantize to Float32</MenuItem>
                            </Select>
                          </FormControl>
                        </>
                      )}

                      {unslothMode === 'gguf_convert' && (
                        <>
                          <Typography variant="h6" gutterBottom>GGUF Conversion</Typography>
                          <Select
                            fullWidth
                            value={ggufInputModel}
                            onChange={(e) => setGgufInputModel(e.target.value)}
                            displayEmpty
                            sx={{ mb: 2 }}
                          >
                            <MenuItem value="">Select Input Model</MenuItem>
                            <ListSubheader>Hugging Face Models</ListSubheader>
                            {huggingfaceFolders.map((folder) => (
                              <MenuItem key={`hf-${folder}`} value={folder}>{folder}</MenuItem>
                            ))}
                            <ListSubheader>Oven Models</ListSubheader>
                            {ovenModels.map((model) => (
                              <MenuItem key={`oven-${model}`} value={model}>{model}</MenuItem>
                            ))}
                          </Select>
                          <TextField
                            fullWidth
                            label="Output Model Name"
                            value={ggufOutputName}
                            onChange={(e) => setGgufOutputName(e.target.value)}
                            sx={{ mb: 2 }}
                          />
                          <Select
                            fullWidth
                            value={ggufOuttype}
                            onChange={(e) => setGgufOuttype(e.target.value)}
                            sx={{ mb: 2 }}
                          >
                            <MenuItem value="q4_0">Q4_0</MenuItem>
                            <MenuItem value="q4_1">Q4_1</MenuItem>
                            <MenuItem value="q5_0">Q5_0</MenuItem>
                            <MenuItem value="q5_1">Q5_1</MenuItem>
                            <MenuItem value="q8_0">Q8_0</MenuItem>
                            <MenuItem value="f16">F16</MenuItem>
                            <MenuItem value="f32">F32</MenuItem>
                          </Select>
                          <Button 
                            fullWidth
                            variant="contained" 
                            onClick={runGgufConversion}
                            disabled={!ggufInputModel}
                            sx={{ mb: 2 }}
                          >
                            Convert to GGUF
                          </Button>
                        </>
                      )}

                      {unslothMode === 'dequantize' && (
                        <>
                          <Typography variant="h6" gutterBottom>Dequantize Model</Typography>
                          <Select
                            fullWidth
                            value={dequantizeInputModel}
                            onChange={(e) => setDequantizeInputModel(e.target.value)}
                            displayEmpty
                            sx={{ mb: 2 }}
                          >
                            <MenuItem value="">Select Input Model</MenuItem>
                            <ListSubheader>Oven Models</ListSubheader>
                            {ovenModels.map((model) => (
                              <MenuItem key={`oven-${model}`} value={model}>{model}</MenuItem>
                            ))}
                          </Select>
                          <TextField
                            fullWidth
                            label="Output Model Name"
                            value={dequantizeOutputName}
                            onChange={(e) => setDequantizeOutputName(e.target.value)}
                            sx={{ mb: 2 }}
                          />
                          <Select
                            fullWidth
                            value={dequantizePrecision}
                            onChange={(e) => setDequantizePrecision(e.target.value)}
                            sx={{ mb: 2 }}
                          >
                            <MenuItem value="f16">Float16</MenuItem>
                            <MenuItem value="f32">Float32</MenuItem>
                            <MenuItem value="8bit">8-bit</MenuItem>
                          </Select>
                          <Button 
                            fullWidth
                            variant="contained" 
                            onClick={runDequantization}
                            disabled={!dequantizeInputModel}
                            sx={{ mb: 2 }}
                          >
                            Dequantize Model
                          </Button>
                        </>
                      )}

                        {unslothMode === 'quantize' && (
                              <>
                                <Typography variant="h6" gutterBottom>Quantize GGUF</Typography>
                                <Select
                                  fullWidth
                                  value={ggufQuantizeInputModel}
                                  onChange={(e) => {
                                    setGgufQuantizeInputModel(e.target.value);
                                    handleGgufFileSelection(e.target.value);
                                  }}
                                  displayEmpty
                                  sx={{ mb: 2 }}
                                >
                                  <MenuItem value="">Select GGUF Model</MenuItem>
                                  {ggufFiles.map((file) => (
                                    <MenuItem key={file} value={file}>{file}</MenuItem>
                                  ))}
                                </Select>
                                {selectedGgufInfo && (
                                  <Typography variant="body2" sx={{ mb: 2 }}>
                                    Current quantization: {selectedGgufInfo.quantization}
                                  </Typography>
                                )}
                                <TextField
                                  fullWidth
                                  label="Output Model Name"
                                  value={ggufQuantizeOutputName}
                                  onChange={(e) => setGgufQuantizeOutputName(e.target.value)}
                                  sx={{ mb: 2 }}
                                />
                                <Select
                                  fullWidth
                                  value={ggufQuantizeType}
                                  onChange={(e) => setGgufQuantizeType(e.target.value)}
                                  sx={{ mb: 2 }}
                                >
                                  <MenuItem value="q4_0">Q4_0</MenuItem>
                                  <MenuItem value="q4_1">Q4_1</MenuItem>
                                  <MenuItem value="q5_0">Q5_0</MenuItem>
                                  <MenuItem value="q5_1">Q5_1</MenuItem>
                                  <MenuItem value="q8_0">Q8_0</MenuItem>
                                  <MenuItem value="f16">F16</MenuItem>
                                  <MenuItem value="f32">F32</MenuItem>
                                </Select>
                                <Button 
                                  fullWidth
                                  variant="contained" 
                                  onClick={runGgufQuantization}
                                  disabled={!ggufQuantizeInputModel || !ggufQuantizeType}
                                  sx={{ mb: 2 }}
                                >
                                  Quantize GGUF
                                </Button>
                              </>
                            )}

                          {(unslothMode === 'train' || unslothMode === 'merge') && (
                            <Button 
                              fullWidth
                              variant="contained" 
                              onClick={unslothMode === 'train' ? runUnslothTraining : runUnslothMerge}
                              disabled={
                                unslothMode === 'train'
                                  ? !selectedTrainingFile || !selectedHuggingfaceModel
                                  : !mergeBaseModel || !mergeAdapterModel || !mergeOutputName
                              }
                              sx={{ mb: 2 }}
                            >
                              {unslothMode === 'train' ? 'Run Unsloth Training' : 'Run Unsloth Merge'}
                            </Button>
                          )}

                          {showTrainingProgress && trainingProgress && (
                            <TrainingProgressVisualization progress={trainingProgress.progress} metrics={trainingProgress.metrics} />
                          )}
                        </Paper>
                      </Grid>
              {error && (
                <Grid item xs={12}>
                  <Paper elevation={3} sx={{ p: 2, mb: 2, bgcolor: 'error.main' }}>
                    <Typography variant="h6" gutterBottom sx={{ color: 'white' }}>Error</Typography>
                    <Typography sx={{ color: 'white' }}>{error}</Typography>
                  </Paper>
                </Grid>
              )}
              <Grid item xs={12}>
                <Paper elevation={3} sx={{ p: 2 }}>
                  <Typography variant="h5" gutterBottom>Output</Typography>
                  <pre>{output}</pre>
                </Paper>
              </Grid>
            </Grid>
          </MainContent>
        </Box>
        <Dialog open={openCustomTemplateDialog} onClose={handleCloseCustomTemplateDialog}>
          <DialogTitle>Add Custom Chat Template</DialogTitle>
          <DialogContent>
            <TextField
              autoFocus
              margin="dense"
              label="Template Name"
              fullWidth
              value={customTemplateName}
              onChange={(e) => setCustomTemplateName(e.target.value)}
            />
            <TextField
              margin="dense"
              label="Template Content"
              fullWidth
              multiline
              rows={10}
              value={customTemplateContent}
              onChange={(e) => setCustomTemplateContent(e.target.value)}
              helperText="Use {SYSTEM}, {INPUT}, and {OUTPUT} placeholders in your template."
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={handleCloseCustomTemplateDialog}>Cancel</Button>
            <Button onClick={handleSaveCustomTemplate}>Save</Button>
          </DialogActions>
        </Dialog>
      </ThemeProvider>
    );
  }
  
  export default App;