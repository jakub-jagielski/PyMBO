# PyMBO Software - Manual Testing Checklist

This checklist provides a detailed set of manual tests to verify the functionality of the PyMBO software.

## 1. Installation and Setup

- [x] **Dependency Check:**
  - [x] Verify all dependencies in `requirements.txt` are installable via `pip install -r requirements.txt`.
  - [x] Check for any version conflicts.
- [x] **Installation from source:**
  - [x] Run `python setup.py install` and check for successful installation.
- [x] **Package Installation:**
  - [x] Test installation from a package manager (if applicable, e.g., `pip install pymbo`).
- [x] **Application Launch:**
  - [x] Launch the GUI via `python main.py`.
  - [x] Launch the GUI via `python -m pymbo.cli`.
  - [x] Check for any console errors on startup.
- [x] **Dependency Error Handling:**
  - [x] Uninstall a required dependency (e.g., `pip uninstall numpy`).
  - [x] Launch the application and verify that a user-friendly error dialog appears, listing the missing dependency.
  - [x] Reinstall the dependency and confirm the application launches correctly.
- [x] **Logging:**
  - [x] Check if the `logs` directory is created on startup.
  - [x] Verify that the `optimization_enhanced.log` file is created and logs initial startup messages.

## 2. GUI - Main Window and General Functionality

- [x] **Welcome Screen:**
  - [x] Verify the application opens to a welcome screen.
  - [x] Check that the title is "Multi-Objective Optimization Laboratory v3.1.5".
  - [x] Test the "Start New Optimization" button.
  - [x] Test the "SGLBO Screening" button.
  - [x] Test the "Load Study" button.
  - [x] Test the "Import Data" button.
- [x] **Window Behavior:**
  - [x] Check that the window can be resized, minimized, and maximized.
  - [x] Verify the minimum window size constraints.
- [x] **Status Bar:**
  - [x] Check for an initial "Ready" status message.
  - [x] Verify the status bar updates during operations (e.g., "Processing...", "View update completed.").

## 3. GUI - Optimization Setup

### 3.1. Parameters Tab

- [x] **Add/Remove Parameters:**
  - [x] Add a new parameter row using the "Add Parameter" button.
  - [x] Remove a parameter row using the "Remove" button.
- [x] **Parameter Types:**
  - [x] Select "continuous" and verify the "Bounds/Values" field shows a format like `[0.0, 100.0]`.
  - [x] Select "discrete" and verify the "Bounds/Values" field shows a format like `[0, 10]`.
  - [x] Select "categorical" and verify the "Bounds/Values" field shows a format like `Value1, Value2, Value3`.
- [x] **Parameter Goals:**
  - [x] Select "Target" as a goal and verify the "Target" entry field becomes enabled.
  - [x] Select "None", "Maximize", or "Minimize" and verify the "Target" field is disabled.
- [x] **Input Validation:**
  - [x] Enter invalid bounds for a continuous parameter (e.g., `[100, 0]`, `[a, b]`) and click "Start Optimization". Verify an error message is shown.
  - [x] Enter invalid values for a categorical parameter (e.g., empty string) and check for errors.
  - [x] Enter a non-numeric target value and check for errors.

### 3.2. Responses Tab

- [x] **Add/Remove Responses:**
  - [x] Add a new response row using the "Add Response" button.
  - [x] Remove a response row using the "Remove" button.
- [x] **Response Goals:**
  - [x] Select "Target" or "Range" as a goal and verify the "Target" entry field becomes enabled.
  - [x] Select "Maximize" or "Minimize" and verify the "Target" field is disabled.
- [x] **Input Validation:**
  - [x] Enter an invalid target/range value (e.g., non-numeric, incorrect range format) and check for errors.

### 3.3. Starting Optimization

- [x] **Initial Sampling Method:**
  - [x] Test starting an optimization with "Random" sampling.
  - [x] Test starting an optimization with "LHS" sampling.
- [x] **Validation on Start:**
  - [x] Try to start an optimization with no parameters defined. Verify error.
  - [x] Try to start an optimization with no responses defined. Verify error.
  - [x] Try to start an optimization with no optimization goals set for any parameter or response. Verify error.

## 4. GUI - Running an Optimization

### 4.1. Next Experiment Tab

- [x] **Initial Suggestion:**
  - [x] Verify that an initial experiment suggestion is displayed immediately after starting an optimization.
- [ ] **Refresh Suggestion:**
  - [ ] Click "Get New Suggestion" and verify a new suggestion is generated.
- [ ] **Batch Suggestions:**
  - [ ] Enter a number (e.g., 5) in the "Number of Suggestions" field and click "Generate Batch".
  - [ ] Verify that the correct number of suggestions are generated and displayed in the text area.
  - [ ] Test with a large number of suggestions (e.g., 50).
- [ ] **Export Batch Suggestions:**
  - [ ] Generate a batch of suggestions.
  - [ ] Click "Download CSV" and save the file.
  - [ ] Open the CSV and verify its format and content. It should contain all parameter and response columns.

### 4.2. Submit Results Tab

- [ ] **Submit Single Result:**
  - [ ] Enter valid numeric results for all responses.
  - [ ] Click "Submit Results".
  - [ ] Verify that the "Next Experiment" tab updates with a new suggestion.
  - [ ] Verify that the plots update.
- [ ] **Input Validation:**
  - [ ] Try to submit non-numeric or empty results. Verify an error message is shown.

### 4.3. Best Solution Tab

- [ ] **Display Update:**
  - [ ] After submitting a few results, check the "Best Solution" tab.
  - [ ] Verify that it displays the best-found parameter values and the corresponding predicted response values.
  - [ ] Check that the confidence intervals are displayed for the responses.

## 5. GUI - Plotting and Visualization

For each plot type, perform the following checks:

- [x] **Pareto Front Plot:**
  - [x] Verify the plot shows all solutions and the Pareto optimal points.
  - [x] Test the objective selectors to plot different objective pairs.
  - [x] Check the plot updates after submitting new results.
  - [x] Test the plot controls (axis ranges, etc.).
- [x] **Progress Plot:**
  - [x] Verify the plot shows the hypervolume trend over iterations.
  - [x] Check for both raw and normalized hypervolume plots.
  - [x] Test the plot controls.
- [x] **GP Slice Plot:**
  - [x] Select different responses and parameters to plot.
  - [x] Use the slider to change the value of the fixed parameter and verify the plot updates.
  - [x] Check that the GP mean and confidence interval are displayed correctly.
  - [x] Test the plot controls.
- [x] **3D Surface Plot:**
  - [x] Select different parameters and a response to plot.
  - [x] Verify the 3D surface is rendered correctly.
  - [x] Test rotating and zooming the 3D plot.
  - [x] Test the plot controls (colormap, style, etc.).
- [x] **Parallel Coordinates Plot:**
  - [x] Verify the plot is generated.
  - [x] Test the variable selection in the plot controls.
  - [x] Check that lines are colored by a response value.
- [x] **GP Uncertainty Map:**
  - [x] Select different parameters and a response.
  - [x] Verify the heatmap/contour shows the uncertainty distribution.
  - [x] Test different uncertainty metrics (std, variance).
  - [x] Test the plot controls.
- [x] **Model Diagnostics Plots (Parity and Residuals):**
  - [x] Select a response and check the parity plot (predicted vs. actual).
  - [x] Check the residuals plot.
  - [x] Verify the plots update with new data.
- [ ] **Sensitivity Analysis Plot:**
  - [ ] Select a response and an algorithm.
  - [ ] Verify the sensitivity analysis plot is generated.
  - [ ] Test all available algorithms (Variance-based, Morris, etc.).
  - [ ] Test the iteration controls and other options in the control panel.

## 6. GUI - SGLBO Screening Module

- [ ] **Launch Screening:**
  - [ ] From the welcome screen, click "SGLBO Screening".
  - [ ] Go through the setup wizard and define parameters/responses.
- [ ] **Interactive Screening Window:**
  - [ ] Verify the interactive screening window appears.
  - [ ] Test submitting results manually and getting the next suggestion.
  - [ ] Check the real-time updates of the screening plots.
- [ ] **Screening Plots:**
  - [ ] **Parameter Space:** Check the plot of experimental points and gradient vectors.
  - [ ] **Response Trends:** Verify the plot of response values over time.
  - [ ] **Parameter Importance:** Check the bar chart of parameter importances.
  - [ ] **Correlation Matrix:** Verify the heatmap of correlations.
- [ ] **Design Space Generation:**
  - [ ] After screening, click "Generate Design Space".
  - [ ] Verify that a set of design points is generated and displayed.
  - [ ] Test exporting the design space.

## 7. File Operations

- [ ] **Save/Load Optimization Study:**
  - [ ] Start a new optimization and add some data.
  - [ ] Click "Save Optimization", save the file.
  - [ ] Close and reopen the application.
  - [ ] Click "Load Study" and open the saved file.
  - [ ] Verify that the entire state is restored (parameters, responses, data, plots).
- [ ] **Import Experimental Data:**
  - [ ] Create a CSV file with experimental data.
  - [ ] Start a new optimization with matching parameters/responses.
  - [ ] Use the "Upload CSV" button in the batch suggestions section to import the data.
  - [ ] Verify the data is added correctly and the plots update.
  - [ ] Test with a CSV file containing errors (e.g., missing columns, non-numeric data) and check for error handling.

## 8. Reporting

- [ ] **Generate Report:**
  - [ ] After running an optimization, click "Generate Report".
  - [ ] Test generating a "Comprehensive" report.
  - [ ] Test generating a "Statistical" report.
- [ ] **Report Formats:**
  - [ ] Generate a report in HTML format and view it in a browser.
  - [ ] Generate a report in PDF format.
  - [ ] Generate a report in Markdown format.
  - [ ] Generate a report in JSON format.
- [ ] **Report Content:**
  - [ ] Verify the report contains the correct summary, parameters, objectives, best solution, and plots.

## 9. Core Optimizer Logic (for programmatic testing)

- [ ] **Instantiate Optimizer:**
  - [ ] Create an instance of `EnhancedMultiObjectiveOptimizer` with a valid configuration.
- [ ] **Suggest and Add Data:**
  - [ ] Call `suggest_next_experiment()` and check the format of the suggestion.
  - [ ] Create a pandas DataFrame with results and call `add_experimental_data()`.
  - [ ] Repeat this process for several iterations.
- [ ] **Get Results:**
  - [ ] Call `get_pareto_front()` and verify the returned DataFrames.
  - [ ] Call `get_best_compromise_solution()` and check the returned dictionaries.
- [ ] **Error Handling:**
  - [ ] Try to instantiate the optimizer with an invalid configuration and check for `ValueError`.

## 10. Performance and Error Handling

- [ ] **Large Datasets:**
  - [ ] Import a large CSV file (e.g., 100+ experiments).
  - [ ] Check for UI responsiveness during and after the import.
  - [ ] Verify that plots handle the large amount of data without freezing.
- [ ] **Error Conditions:**
  - [ ] Try to perform actions in the wrong order (e.g., submit results before starting an optimization).
  - [ ] Check for clear and informative error messages.
- [ ] **Log File Check:**
  - [ ] After a testing session, review the `optimization_enhanced.log` file for any unexpected errors or warnings.
