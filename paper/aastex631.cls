\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{aastex631}[2026/06/08 local compatibility shim for AASTeX 7.0.1]
\LoadClass{aastex701}
\makeatletter
\def\email@error#1{\ClassWarning{aastex631}{AASTeX 7.0.1 requires an email address for author "#1"; continuing for local draft compilation}}
\makeatother
