
;; @We define the package <<cl-ann.matlab-utils>> to provide utilities that ease
;; the interaction between our Common Lisp framework and the Matlab/Octave based
;; resources provided by the class organizers.

;; @\section{Parsing Data Files}
;; \label{parsing}

;; <<>>=
(defpackage :cl-ann.matlab-utils
  (:use :cl :iterate :index-mapped-arrays :alexandria :cl-ppcre)
  (:export
   #:read-mat-data-file))

(in-package :cl-ann.matlab-utils)

;; @We can process a simple octave/matlab array data when specified in ascii
;; format.  These are the files that are provided via the files like
;; 'dataset1_ancient_octave.mat'.  We will not bother to work out the way to
;; read the new binary '.mat' files, but we may revisit this in the future.

;; You can use <<read-mat-data-file>> to parse a data file.  This will return an
;; alist with the following symbol values: <:w-init>, <:w-gen-feas>, and
;; <:examples-nobias>.  The positive and negative examples are combined into one
;; vector and marked with a <t> or <nil> value as their first element as this
;; seems much cleaner.

;;<<>>=
(defun read-mat-data-file (file)
  "Take a file and return an alist of data values from the file.  The positive
and negative examples are combined and simply marked by a boolean at the
beginning.  The symbols in the alist should be :w-init, :w-gen-feas,
and :examples-nobias."
  (let* ((dataset (%read-mat-data-file file))
         training-data
         (data (iter (for (field value) in dataset)
                 (cond ((member field '(:w-init :w-gen-feas))
                        (collect (list field (mapcar 'first value))))
                       ((eql :pos-examples-nobias field)
                        (setf training-data
                              (append training-data
                                      (mapcar (lambda (x) (cons t x)) value))))
                       ((eql :neg-examples-nobias field)
                        (setf training-data
                              (append training-data
                                      (mapcar (lambda (x) (cons nil x)) value))))
                       (t (collect (list field value)))))))
    (cons (list :examples-nobias training-data) data)))

;;<<>>=
(defun %read-mat-data-file (file)
  "The internal worker function for parsing the 'ancient Octave MAT files'."
  (let ((file (format nil "~{~A~%~}"
                      (iter (for line in-file file using 'read-line)
                        (collect line)))))
    (iter (with start = 0)
      (for st = (scan "# name:" file :start start))
      (while st)
      (for name = (tb:reg-scan-to-string "# name:\\s*(\\S*)\\s*" file :start st))
      (collecting
       (list
        (intern (string-upcase (regex-replace-all "_" name "-"))
                :keyword)
        (iter (for line in-stream (make-string-input-stream file st) using 'read-line)
          (until (scan "^\\s*$" line))
          (unless (scan "\\s*#" line)
            (collect (iter (for val in-stream (make-string-input-stream line))
                       (collect val)))))))
      (setf start (1+ st)))))

;; @\section{Perceptrons}

;;<<>>=
(defpackage :cl-ann.perceptron
  (:use :cl :iterate :index-mapped-arrays :alexandria
        :zgnuplot
        :cl-ann.matlab-utils)
  (:export))

(in-package :cl-ann.perceptron)

;; @In this section we define the code (well code with pieces missing) for
;; training and predicting using perceptrons.  This basic structure of the
;; program is broken into __ pieces: The prediction procedure <<predict>>, the
;; training framework <<train-perception>>, the weight adjuster <<new-weights>>,
;; and the perceptron correctness validator <<errors-in-training-data>>.  In
;; this system, <<train-perceptron>> will iterate until
;; <<errors-in-training-data>> confirms the perceptron is correct (or otherwise
;; instructed to stop) updating the perceptron weights via <<new-weights>>.
;; Your task is to write <<new-weights>>.

;; @We read in the data provided as part of the assignment in <<datasets>>.  See
;; the section \ref{parsing} on parsing the data format for a better
;; understanding on how this is processed.

;;<<datasets,4>>=
(defparameter *dataset1* (read-mat-data-file #p"dat/dataset1_ancient_octave.mat"))
(defparameter *dataset2* (read-mat-data-file #p"dat/dataset2_ancient_octave.mat"))
(defparameter *dataset3* (read-mat-data-file #p"dat/dataset3_ancient_octave.mat"))
(defparameter *dataset4* (read-mat-data-file #p"dat/dataset4_ancient_octave.mat"))

;; @The math of determining the activity of a perceptron is simplified by
;; appending the bias, an input which is always 1, to the feature vectors of the
;; example cases.  The <<append-bias>> function does just that.

;;<<>>=
(defun append-bias (training-case)
  "Append the bias to the end of each training case's feature vector."
  (append training-case (list 1)))

;; @The <<predict>> function performs the calculaion $w\cdot x$ where $w$ is the
;; is the weights and $x$ is the feature vector with the bias appended.

;;<<>>=
(defun predict (weights inputs)
  "Perform a prediction using the list of perceptron weights and a list of
feature inputs."
  (<= 0 (iterate
          (for w in weights)
          (for i in (append-bias inputs))
          (summing (* w i)))))

;; @The <<new-weights>> function takes some current perceptron weights and a set
;; of training data and returns new weights that should be closer to a feasible
;; set, i.e.\ weights that will predict all of the training data.  This function
;; is used as an input to the <<train-perceptron>> function.

;; @You should write/edit the <<new-weights>> function.  The implementation here
;; has some structure that should get you started, but the rest is up to you.
;; The idea is that it will take a list of initial weights and a list of
;; training data each of the form {\texttt (target-t-or-nil . feature-vector)}
;; and it train the perceptron on that data and return the new trained weight
;; for that system (the last weight corresponds to the bias weight by convention
;; in the provided MatLab/Otave support code).  The <<new-weights>> function
;; also takes a keyword parameter <batch> which if true should instruct the
;; procedure to train in batch mode.  It should be noted that the original task
;; for the class requires you to implement `on-line' learning and you will need
;; to do so to complete the assignment.  Feel free to ignore the batch parameter
;; if you wish.

;;<<>>=
(defun new-weights (initial-weights training-data
                    &key batch)
  "Take a list of initial-weights and a list of training data examples and
return a new set of weights.  The parameter batch requests that learning should
be done using all training data to update the weights once, rather than via an
`on-line' method."
  (let ((training-data (mapcar 'append-bias training-data)))
    (iter (for (target . training-vector) in training-data)
      (let ((prediction (predict initial-weights training-vector)))
        (when on-iterate (funcall on-iterate target prediction training-vector))
        (cond ((and target (not prediction))
               ;; Your code here
               )
              ((and (not target) prediction)
               ;; Your code here
               )))))
  initial-weights)

;; @The <<train-perceptron>> function provides a mechanism to repeatedly call
;; the <<new-weights>> function and detect when it has completed its training,
;; i.e.\ the weight vector correctly predicts all training data.  It can also
;; produce output detailing the training procedure in text, graphically, or both
;; using the <progress-reports> option.

;;<<>>=
(defun train-perceptron (data initial-weights
                         &key (training-function 'new-weights)
                              max-iterations
                              progress-reports)
  "Train the perceptron given data and initial weights.  The training function,
the function that gives new weights, can be specified via the training-function
parameter.  You may also set a maximum number of iterations before the training
procedure gives up.  The progress-reports parameter allows you to specify how
much training output you want.  This should be specified as a keyword symbol of
the form :print, :plot, :confirm or a list containing one or more of these
options.

  :print will output a small progress report per iteration that will tell how many
  errors are still present.

  :plot will plot a summary of the classification per iteration (including a
  separating line for the perceptron).

  :confirm will break the procedure at each iteration."
  (let ((progress-reports (alexandria:ensure-list progress-reports)))
    (iter
      (for i from 0)
      (when (and max-iterations (> i max-iterations))
        (error "No solution found"))
      (for w-trained
        initially initial-weights
        then (funcall training-function w-trained data))
      (let ((errors (errors-in-training-data w-trained data)))
        (when (member :print progress-reports)
          (format t "~%Iteration ~A weights: ~A" i w-trained)
          (apply 'print-error-report errors))
        (when (member :plot progress-reports)
          (plot-perceptron w-trained data))
        (when (member :confirm progress-reports)
          (break))
        (while (some 'identity errors))
        (finally (return w-trained))))))

;; @The <<errors-in-training-data>> function collects the two types of errors,
;; false positives and false negatives, that the perceptron makes using the
;; given weights and the example data.  The function <<print-error-report>> can
;; be used to print this data in a more human readable format.

;;<<>>=
(defun errors-in-training-data (weights data)
  "Find with examples are mis-classified.  Returns a list of two lists, the
false-positives and the false-negatives."
  (let ((data (mapcar 'append-bias data)))
    (iter (for (target . training-vector) in data)
      (let ((prediction (predict weights training-vector)))
        (cond ((and target (not prediction))
               (collecting training-vector into false-negatives))
              ((and (not target) prediction)
               (collecting training-vector into false-positives))))
      (finally (return (list false-positives false-negatives))))))

;;<<>>=
(defun print-error-report (false-positives false-negatives)
  "Print an error report using the output of errors-in-training-data."
  (format *error-output* "~%Errors:~
                          ~%  False positives: ~A~
                          ~%  False negatives: ~A"
          (length false-positives) (length false-negatives)))

;; @\section{Plotting Perceptron Performance}

;; @We define some plotting procedures using the ZGNUPlot.  This doesn't include
;; code for colorizing mis-categorized examples, nor does it plot the other data
;; (the number of errors or the distance to the generously feasible solution)
;; that the provided code does.

;;<<>>=
(defun plot-perceptron (weights training-data)
  "Plot the tests by their classification and the line that the perceptron is
attempting to use to separate them."
  (plot
   (make-gnuplot-setup :view-metric-equivalence t)
   (destructuring-bind (wx wy wbias) weights
     (let* ((weight-vec (list wx wy))
            (norm (iter (for el in weight-vec)
                    (summing (expt el 2))))
            (norm-weight-vec (iter (for el in weight-vec)
                               (collect (/ el norm))))
            (root (mapcar (lambda (x) (* (- wbias) x)) norm-weight-vec)))
       (let ((y-intercept
               (- (second root)
                  (* (first root)
                     (/ (- (first weight-vec)) (second weight-vec))))))
         (lambda (x) (+ (* x (/ (- (first weight-vec)) (second weight-vec)))
                   y-intercept)))))
   (mapcar 'rest (remove-if 'first training-data))
   (mapcar 'rest (remove-if-not 'first training-data))))
