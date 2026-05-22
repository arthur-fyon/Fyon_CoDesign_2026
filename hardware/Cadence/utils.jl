using CSV, DataFrames, Plots, DelimitedFiles

"""
    parse_vcsv(filename::String)

Parse a VCSV file from Virtuoso Cadence and return the data as a DataFrame.

The function handles:
- Header lines starting with ';'
- Multiple data sections separated by commas in header
- Automatic column naming based on header information
- Proper parsing of scientific notation
"""
function parse_vcsv(filename::String)
    lines = readlines(filename)

    # Find header information
    header_lines = filter(line -> startswith(line, ';'), lines)
    data_lines = filter(line -> !startswith(line, ';'), lines)

    # Parse header to understand data structure
    version_line = findfirst(line -> contains(line, "Version"), header_lines)
    if version_line !== nothing
        println("VCSV Version: ", split(header_lines[version_line], ',')[2:end])
    end

    # Find variable names and units
    var_names = String[]
    var_units = String[]

    for line in header_lines
        if contains(line, "X, Y") || contains(line, "Re, Re") || contains(line, "Re, Im")
            # Extract variable names from lines like ";X, Y,;X, Y"
            parts = split(line, ';')
            for part in parts
                if !isempty(strip(part))
                    vars = split(part, ',')
                    for var in vars
                        var_clean = strip(var)
                        if !isempty(var_clean) && !(var_clean in var_names)
                            push!(var_names, var_clean)
                        end
                    end
                end
            end
        end
    end

    # If no variable names found, create default ones
    if isempty(var_names)
        # Count columns from first data line
        if !isempty(data_lines)
            ncols = length(split(data_lines[1], ','))
            var_names = ["col$i" for i in 1:ncols]
        end
    end

    # Parse data
    if isempty(data_lines)
        return DataFrame()
    end

    # Convert data lines to matrix
    data_matrix = []
    for line in data_lines
        if !isempty(strip(line))
            parts = split(line, ',')
            row = [let s = strip(p); isempty(s) ? NaN : parse(Float64, s) end for p in parts]
            push!(data_matrix, row)
        end
    end

    # Convert to DataFrame
    if !isempty(data_matrix)
        # Ensure all rows have the same length
        max_cols = maximum(length.(data_matrix))
        data_matrix = [length(row) == max_cols ? row : [row; zeros(max_cols - length(row))] for row in data_matrix]

        # Create DataFrame
        df = DataFrame(reduce(hcat, data_matrix)', :auto)

        # Rename columns if we have variable names
        if length(var_names) >= ncol(df)
            rename!(df, Symbol.(var_names[1:ncol(df)]))
        elseif length(var_names) > 0
            # Use available names and generate the rest
            col_names = [var_names; ["col$(i)" for i in (length(var_names)+1):ncol(df)]]
            rename!(df, Symbol.(col_names))
        end

        return df
    else
        return DataFrame()
    end
end

"""
    plot_vcsv_data(df::DataFrame, filename::String;
                   x_col=1, y_col=2, title="", xlabel="", ylabel="")

Create a plot from VCSV data DataFrame.
"""
function plot_vcsv_data(df::DataFrame, filename::String;
                       x_col=1, y_col=2, title="", xlabel="", ylabel="",
                       plot_type=:line, kwargs...)

    if isempty(df)
        @warn "DataFrame is empty"
        return nothing
    end

    x_data = df[!, x_col]
    y_data = df[!, y_col]

    # Create appropriate title if not provided
    if isempty(title)
        title = "Data from $(basename(filename))"
    end

    # Create labels if not provided
    if isempty(xlabel)
        xlabel = string(names(df)[x_col])
    end
    if isempty(ylabel)
        ylabel = string(names(df)[y_col])
    end

    if plot_type == :line
        p = plot(x_data, y_data,
                title=title, xlabel=xlabel, ylabel=ylabel,
                linewidth=2, legend=false; kwargs...)
    elseif plot_type == :scatter
        p = scatter(x_data, y_data,
                   title=title, xlabel=xlabel, ylabel=ylabel,
                   legend=false; kwargs...)
    else
        p = plot(x_data, y_data,
                title=title, xlabel=xlabel, ylabel=ylabel,
                seriestype=plot_type, legend=false; kwargs...)
    end

    return p
end

"""
    analyze_vcsv_file(filename::String)

Analyze a VCSV file and provide summary information.
"""
function analyze_vcsv_file(filename::String)
    println("Analyzing VCSV file: $filename")
    println("="^50)

    # Read and parse the file
    df = parse_vcsv(filename)

    if isempty(df)
        println("No data found in file")
        return nothing
    end

    println("Data dimensions: $(size(df))")
    println("Column names: $(names(df))")
    println()

    # Show basic statistics
    println("Data summary:")
    println(describe(df))
    println()

    # Show first few rows
    println("First 5 rows:")
    println(first(df, 5))
    println()

    return df
end

# Example usage functions for your specific files

"""
    get_bistable_transient_data(filename::String)

Extract x,y data from bistable cell transient simulation.
Returns (x_data, y_data) tuple.
"""
function get_bistable_transient_data(filename::String)
    df = parse_vcsv(filename)
    if isempty(df)
        return nothing, nothing
    end

    # Assuming time is first column and current is second
    time_col = 1
    current_col = 2

    x_data = df[!, time_col]
    y_data = df[!, current_col]

    return x_data, y_data
end

"""
    get_parameter_sweep_data(filename::String)

Extract x,y data from parameter sweep.
Returns (x_data, y_data) tuple.
"""
function get_parameter_sweep_data(filename::String)
    df = parse_vcsv(filename)
    if isempty(df)
        return nothing, nothing
    end

    x_data = df[!, 1]  # First column
    y_data = df[!, 2]  # Second column

    return x_data, y_data
end

"""
    get_xy_data(filename::String, x_col=1, y_col=2)

Extract x,y data from any VCSV file.
Returns (x_data, y_data) tuple.
"""
function get_xy_data(filename::String, x_col=1, y_col=2)
    df = parse_vcsv(filename)
    if isempty(df)
        return nothing, nothing
    end

    x_data = df[!, x_col]
    y_data = df[!, y_col]

    return x_data, y_data
end

"""
    get_all_xy_pairs(filename::String)

Extract all x,y pairs from a VCSV file with multiple curves.
Returns array of (x_data, y_data) tuples.
"""
function get_all_xy_pairs(filename::String)
    df = parse_vcsv(filename)
    if isempty(df)
        return []
    end

    xy_pairs = []
    ncols = ncol(df)

    # Extract pairs assuming columns alternate between x and y data
    for i in 1:2:ncols-1
        if i+1 <= ncols
            x_data = df[!, i]
            y_data = df[!, i+1]

            # Filter out any NaN values
            valid_idx = .!isnan.(x_data) .& .!isnan.(y_data)
            if any(valid_idx)
                push!(xy_pairs, (x_data[valid_idx], y_data[valid_idx]))
            end
        end
    end

    return xy_pairs
end

"""
    get_column_data(filename::String, col_idx::Int)

Extract data from a specific column.
Returns the column data as a vector.
"""
function get_column_data(filename::String, col_idx::Int)
    df = parse_vcsv(filename)
    if isempty(df) || col_idx > ncol(df)
        return nothing
    end

    return df[!, col_idx]
end

"""
    get_all_columns(filename::String)

Extract all columns from a VCSV file.
Returns array of column vectors.
"""
function get_all_columns(filename::String)
    df = parse_vcsv(filename)
    if isempty(df)
        return []
    end

    columns = []
    for i in 1:ncol(df)
        push!(columns, df[!, i])
    end

    return columns
end

# Example usage with your files
println("VCSV Reader for Virtuoso Cadence - Ready!")
println("Available functions:")
println("- parse_vcsv(filename): Parse VCSV file to DataFrame")
println("- analyze_vcsv_file(filename): Analyze file and show summary")
println("- get_xy_data(filename, x_col, y_col): Get x,y data from specific columns")
println("- get_bistable_transient_data(filename): Get time,current data")
println("- get_parameter_sweep_data(filename): Get parameter sweep x,y data")
println("- get_all_xy_pairs(filename): Get all x,y pairs from multiple curves")
println("- get_column_data(filename, col_idx): Get specific column data")
println("- get_all_columns(filename): Get all columns as array")
println()
println("Example usage:")
println("# Get x,y data from transient simulation")
println("time, current = get_bistable_transient_data(\"bistable_cell_trans_triangle.vcsv\")")
println()
println("# Get parameter sweep data")
println("param_vals, outputs = get_parameter_sweep_data(\"bistable_cell_Ithresh_sweep.vcsv\")")
println()
println("# Get all curves from multi-curve file")
println("all_curves = get_all_xy_pairs(\"your_200_curves_file.vcsv\")")
println("# Access first curve: x1, y1 = all_curves[1]")
println()
println("# Get specific columns")
println("x_data, y_data = get_xy_data(\"filename.vcsv\", 1, 2)  # columns 1 and 2")
println("col3_data = get_column_data(\"filename.vcsv\", 3)      # just column 3")
