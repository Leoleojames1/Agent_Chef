import React, { useState, useEffect } from 'react';
import { TextField, Button, Typography, Box, Accordion, AccordionSummary, AccordionDetails, Grid, Select, MenuItem } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import axios from 'axios';

const defaultPrompts = {
  system: `You are helpful function calling dataset construction assistant, please help me build this function command calling dataset for Ollama Agent Roll Cage. You will be given user requests and assistant answers to generate alternative paraphrasing and rephrasing. Maintain all of the details of the description given.`,
  dynamicColumns: {
    input: {
      system: `You are an AI assistant specializing in generating input question data. Your task is to maintain all of the details of the description given, maintaining its original meaning and incorporating the provided reference values. Do not add any explanatory text or meta-information. You will keep all questions as questions and just re-ask them. Please make sure to change up the Wh-question words (What, When, Where, Who, Why, How, Can, Could, Would, Should, Is, Are, Do, Does) and rephrase the questions, you like to repeat the same responses I want variance.`,
      user: `Original text: {text}
Reference values: {reference_values}

Please generate a new input question, maintaining its core meaning and incorporating the reference values where appropriate. Ensure the generated question is coherent and contextually relevant. Provide only the generated question without any additional explanations or formatting.`
    },
    output: {
      system: `You are an AI assistant specializing in generating output statement data. Your task is to maintain all of the details of the description given, maintaining its original meaning and incorporating the provided reference values. Do not add any explanatory text or meta-information. You will keep all statements as statements and just re-state them. Please make sure to change up the statement and rephrase it a different way, you like to repeat the same responses I want variance.  Make sure to correctly incorporate the reference values and do not contain the {text} or {command} format string this is an artifact format.`,
      user: `Original text: {text}
Reference values: {reference_values}

Please generate a new output statement, maintaining its core meaning and incorporating the reference values , rephrasing and describing how to do the task described. Ensure the generated statement is coherent and contextually relevant. Provide only the generated statement without any additional explanations or formatting.`
    }
  },
  verify: {
    system: `You are a verification assistant for Agent Chef, a dataset constructor tool. Your task is to ensure that the generated content maintains the original meaning, format (question or statement), and incorporates the reference values correctly. If the generated content is accurate, return it as-is. If not, provide a corrected version.`,
    user: `Original: {original}
Generated: {generated}
Reference values: {reference}
Is question: {is_question}

Verify that the generated content maintains the original meaning, format (question or statement), and correctly incorporates the reference values and does not contain the {text} or {command} format string this is an artifact format. If it does, return the generated content. If not, provide a corrected version that accurately reflects the original meaning, format, and includes the reference values.
Do not include any explanatory text or meta-information in your response, instead just utilize the existing meaning.`
  }
};

const CustomPromptEditor = ({ columnTypes, onSavePrompts, initialCustomPrompts = {} }) => {
  const [customPrompts, setCustomPrompts] = useState(() => ({
    ...defaultPrompts,
    ...initialCustomPrompts,
    dynamicColumns: {
      ...defaultPrompts.dynamicColumns,
      ...initialCustomPrompts.dynamicColumns
    }
  }));
  const [isUsingDefaults, setIsUsingDefaults] = useState(Object.keys(initialCustomPrompts).length === 0);
  const [promptSetName, setPromptSetName] = useState('');
  const [availablePromptSets, setAvailablePromptSets] = useState([]);

  useEffect(() => {
    fetchPromptSets();
  }, []);

  useEffect(() => {
    const dynamicColumnPrompts = { ...customPrompts.dynamicColumns };
    Object.entries(columnTypes).forEach(([column, type]) => {
      if (type === 'dynamic' && !dynamicColumnPrompts[column]) {
        dynamicColumnPrompts[column] = {
          system: `You are an AI assistant specializing in generating ${column} data.`,
          user: `Given the context and reference values, generate a suitable ${column} response:`
        };
      }
    });
    setCustomPrompts(prev => ({
      ...prev,
      dynamicColumns: dynamicColumnPrompts
    }));
  }, [columnTypes, customPrompts.dynamicColumns]);

  useEffect(() => {
    onSavePrompts(customPrompts);
  }, [customPrompts, onSavePrompts]);

  const fetchPromptSets = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/list_prompt_sets');
      setAvailablePromptSets(response.data);
    } catch (error) {
      console.error('Error fetching prompt sets:', error);
    }
  };

  const handlePromptChange = (category, field, value) => {
    const updatedPrompts = {
      ...customPrompts,
      [category]: field ? { ...customPrompts[category], [field]: value } : value
    };
    setCustomPrompts(updatedPrompts);
    setIsUsingDefaults(false);
    onSavePrompts(updatedPrompts);
  };

  const handleDynamicColumnPromptChange = (column, field, value) => {
    const updatedPrompts = {
      ...customPrompts,
      dynamicColumns: {
        ...customPrompts.dynamicColumns,
        [column]: { ...customPrompts.dynamicColumns?.[column], [field]: value }
      }
    };
    setCustomPrompts(updatedPrompts);
    setIsUsingDefaults(false);
    onSavePrompts(updatedPrompts);
  };

  const handleSave = async () => {
    if (!promptSetName) {
      alert('Please enter a name for the prompt set');
      return;
    }
    try {
      await axios.post('http://localhost:5000/api/save_prompt_set', {
        name: promptSetName,
        prompts: customPrompts
      });
      alert('Prompt set saved successfully');
      fetchPromptSets();
      onSavePrompts(customPrompts);
    } catch (error) {
      console.error('Error saving prompt set:', error);
      alert('Error saving prompt set');
    }
  };

  const handleLoad = async (name) => {
    try {
      const response = await axios.get(`http://localhost:5000/api/load_prompt_set/${name}`);
      setCustomPrompts(response.data);
      setIsUsingDefaults(false);
      setPromptSetName(name);
      onSavePrompts(response.data);
    } catch (error) {
      console.error('Error loading prompt set:', error);
      alert('Error loading prompt set');
    }
  };

  const handleReset = () => {
    setCustomPrompts({ ...defaultPrompts });
    setIsUsingDefaults(true);
    setPromptSetName('');
    onSavePrompts({ ...defaultPrompts });
  };

  const renderPromptField = (label, value, onChange, gridProps = {}) => (
    <Grid item xs={12} {...gridProps}>
      <TextField
        fullWidth
        multiline
        rows={8}
        label={label}
        value={value}
        onChange={onChange}
        variant="outlined"
        margin="normal"
        InputProps={{
          style: { fontFamily: 'monospace', fontSize: '0.9rem' }
        }}
      />
    </Grid>
  );

  return (
    <Box sx={{ mt: 2 }}>
      <Typography variant="h6" gutterBottom>Custom Prompt Editor</Typography>
      <Typography variant="body2" gutterBottom color="textSecondary">
        {isUsingDefaults ? "Using default prompts. Edit to customize." : "Using custom prompts."}
      </Typography>
      
      <Box sx={{ mb: 2 }}>
        <TextField
          label="Prompt Set Name"
          value={promptSetName}
          onChange={(e) => setPromptSetName(e.target.value)}
          sx={{ mr: 2 }}
        />
        <Button variant="contained" color="primary" onClick={handleSave}>
          Save Prompt Set
        </Button>
      </Box>

      <Box sx={{ mb: 2 }}>
        <Select
          value=""
          displayEmpty
          onChange={(e) => handleLoad(e.target.value)}
          sx={{ minWidth: 200, mr: 2 }}
        >
          <MenuItem value="" disabled>Load Prompt Set</MenuItem>
          {availablePromptSets.map(name => (
            <MenuItem key={name} value={name}>{name}</MenuItem>
          ))}
        </Select>
        <Button variant="outlined" color="secondary" onClick={handleReset}>
          Reset to Defaults
        </Button>
      </Box>

      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography>System Prompt</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            {renderPromptField(
              "System Prompt",
              customPrompts.system,
              (e) => handlePromptChange('system', null, e.target.value),
              { xs: 12 }
            )}
          </Grid>
        </AccordionDetails>
      </Accordion>

      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography>Input Column Prompts</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            {renderPromptField(
              "Input System Prompt",
              customPrompts.dynamicColumns.input.system,
              (e) => handleDynamicColumnPromptChange('input', 'system', e.target.value)
            )}
            {renderPromptField(
              "Input User Prompt",
              customPrompts.dynamicColumns.input.user,
              (e) => handleDynamicColumnPromptChange('input', 'user', e.target.value)
            )}
          </Grid>
        </AccordionDetails>
      </Accordion>

      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography>Output Column Prompts</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            {renderPromptField(
              "Output System Prompt",
              customPrompts.dynamicColumns.output.system,
              (e) => handleDynamicColumnPromptChange('output', 'system', e.target.value)
            )}
            {renderPromptField(
              "Output User Prompt",
              customPrompts.dynamicColumns.output.user,
              (e) => handleDynamicColumnPromptChange('output', 'user', e.target.value)
            )}
          </Grid>
        </AccordionDetails>
      </Accordion>

      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography>Verification Prompts</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            {renderPromptField(
              "Verification System Prompt",
              customPrompts.verify.system,
              (e) => handlePromptChange('verify', 'system', e.target.value)
            )}
            {renderPromptField(
              "Verification User Prompt",
              customPrompts.verify.user,
              (e) => handlePromptChange('verify', 'user', e.target.value)
            )}
          </Grid>
        </AccordionDetails>
      </Accordion>

      {Object.entries(columnTypes)
        .filter(([column, type]) => type === 'dynamic' && !['input', 'output'].includes(column))
        .map(([column]) => (
          <Accordion key={column}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography>{column} Prompts</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                {renderPromptField(
                  `${column} System Prompt`,
                  customPrompts.dynamicColumns[column]?.system || '',
                  (e) => handleDynamicColumnPromptChange(column, 'system', e.target.value)
                )}
                {renderPromptField(
                  `${column} User Prompt`,
                  customPrompts.dynamicColumns[column]?.user || '',
                  (e) => handleDynamicColumnPromptChange(column, 'user', e.target.value)
                )}
              </Grid>
            </AccordionDetails>
          </Accordion>
        ))}
    </Box>
  );
};

export default CustomPromptEditor;