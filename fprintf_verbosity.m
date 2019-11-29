function fprintf_verbosity (varargin)
    verbosity = get_verbosity();
    
    if verbosity >= 1
      fprintf(varargin{:})
    end
    
end
