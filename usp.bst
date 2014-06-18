%% usp.bst
%% Copyright 2012 C. C. Fattori
%
% This work may be distributed and/or modified under the
% conditions of the LaTeX Project Public License, either version 1.3
% of this license or (at your option) any later version.
% The latest version of this license is in
%   http://www.latex-project.org/lppl.txt
% and version 1.3 or later is part of all distributions of LaTeX
% version 2005/12/01 or later.
%
% This work has the LPPL maintenance status `maintained'.
% 
% The Current Maintainer of this work is C. C. Fattori.
%
% This work consists of the files usp.cls, uspfemtas.sty, uspbiblio.sty, uspelempt.sty, usptex.layout and usp.bst
%%
ENTRY
  { %standard bibtex entries 
    address
    author
    booktitle 
    chapter	
	day
    edition
    editor 
    journal
	journal-sub
    month 
    note
    number
    organization
    pages
	numpages
    publisher 
    school
    series 
    title
    type
    volume 
	volumes
    year
    booksubtitle
    conference-number 
    conference-year 
    conference-location
    dimensions 
    subtitle 
    year-presented
	compilador
	entidade
	entidade-sub
	doi
	evento
	periodo
	local-publicacao
	jurisdiction
	tradutor
	ilustrador
	revisor
	genero
	depoimento
	forma
	direcao
	inauthor
	ineditor
	inorganization
	incompilador
	inentidade
	inentidade-sub
    }{}{ label extra.label year.label}

%general purpose strings
STRINGS { r s t caractere}

%strings use up a lot of buffer space, we can use only a few
STRINGS {enfase previous.author previous.title}

%most options are implemented with integers
INTEGERS {usp.max.nomes}

%needed for title formatting
INTEGERS { title.lenght title.pos first.word no.bracket nao.comando letra escolhido protegido brackets.abertos tamanho.string primeira.letra posicao.substituicao month.first}
STRINGS { title.char}

%needed for controlling the sentence position
INTEGERS {output.state before.all mid.sentence after.sentence after.block }

FUNCTION {init.state.consts}
{ #0 'before.all :=
  #1 'mid.sentence :=
  #2 'after.sentence :=
  #3 'after.block :=
}


%output.nonnull decides what is written before the argument s, if a comma, period etc.
FUNCTION {output.nonnull}
{ 's :=                                     %the argument is written into s
  output.state mid.sentence =
    { ", " * write$ }                       %mid.sentence first write comma
    { output.state after.block =
        { add.period$ write$ newline$ }
        { output.state before.all = 'write$
            { add.period$ " " * write$ }
          if$}
      if$
      mid.sentence 'output.state := }
  if$
  s                                         %now comes s
}
FUNCTION {output.nonnull.dash}
{ 's :=                                   %the argument is written into s
  output.state mid.sentence =
    { " --- " * write$ }                       %mid.sentence first write comma
    { output.state after.block =
        { add.period$ write$ newline$ write$}
        { output.state before.all = 'write$
            { add.period$ " " * write$ }
          if$}
      if$
      mid.sentence 'output.state := }
  if$
  s                                         %now comes s
}
FUNCTION {output}     %does nothing or calls output.nonull
{ duplicate$ empty$
    'pop$
    'output.nonnull
  if$
}
FUNCTION {output.dash}     %does nothing or calls output.nonull
{ duplicate$ empty$
    'pop$
    'output.nonnull.dash
  if$
}
FUNCTION {output.check}  %like ouput, but complains if t empty
{ 't :=
  duplicate$ empty$
    { pop$ "campo " t * " vazio em " * cite$ * warning$ }
    'output.nonnull
  if$
}

FUNCTION {new.block}
{ output.state before.all =
    'skip$
    { after.block 'output.state := }
  if$
}
FUNCTION {new.sentence}
{ output.state after.block =
    'skip$
    { output.state before.all =
        'skip$
        { after.sentence 'output.state := }
      if$
    }
  if$
}
FUNCTION {add.blank} {  " " * before.all 'output.state :=}

FUNCTION {date.block} {new.block}

FUNCTION {not}
{   { #0 }
    { #1 }
  if$
}
FUNCTION {and}
{   'skip$
    { pop$ #0 }
  if$
}
FUNCTION {or}
{   { pop$ #1 }
    'skip$
  if$
}
FUNCTION {new.block.checka}
{ empty$
    'skip$
    'new.block
  if$
}
FUNCTION {new.block.checkb}
{ empty$
  swap$ empty$
  and
    'skip$
    'new.block
  if$
}
FUNCTION {new.sentence.checka}
{ empty$
    'skip$
    'new.sentence
  if$
}
FUNCTION {new.sentence.checkb}
{ empty$
  swap$ empty$
  and
    'skip$
    'new.sentence
  if$
}
FUNCTION {field.or.null}
{ duplicate$ empty$
    { pop$ "" }
    'skip$
  if$
}
FUNCTION {emphasize}
{ duplicate$ empty$
    { pop$ "" }
    { enfase "{"  *  swap$ * "}" * }
  if$
}

FUNCTION {tie.or.space.prefix} %if text length less than 3 chars. uses a tie
{ duplicate$ text.length$ #5 <
    { "~" }
    { " " }
  if$
  swap$
}

FUNCTION {no.space} { { "" } swap$ }


FUNCTION {space.word} { " " swap$ * " " * }

FUNCTION {bbl.and} {" e "}
FUNCTION {bbl.chapter} { "cap." }
FUNCTION {bbl.colon} {": "}
FUNCTION {bbl.editors} { "Ed." }
FUNCTION {bbl.editor} { "Ed." }
FUNCTION {bbl.edby} { "editado por" }
FUNCTION {bbl.edition} { "ed." }
FUNCTION {bbl.empty} {""}
FUNCTION {bbl.etal}{" et al."}
FUNCTION {bbl.in}{ "In" bbl.colon *}
FUNCTION {bbl.ldots} {"\ldots"}
FUNCTION {bbl.master}  { "Mestrado" }
FUNCTION {bbl.missing.year}{""}
FUNCTION {bbl.monography} { "Monografia" }
FUNCTION {bbl.mthesis} { "Disserta\c{c}\~ao" }
FUNCTION {bbl.no.address} {"[S.l.]"}
FUNCTION {bbl.no.address.no.publisher} {"[S.l.: s.n.]"}
FUNCTION {bbl.no.publisher} {"[s.n.]"}
FUNCTION {bbl.number} { "n." }
FUNCTION {bbl.nr} { "n." }
FUNCTION {bbl.of} { "de" }
FUNCTION {bbl.phd} { "Doutorado" }
FUNCTION {bbl.phdthesis} { "Tese" }
FUNCTION {bbl.pages} { "p." }
FUNCTION {bbl.page} { bbl.pages }
FUNCTION {bbl.reprint} {"Separata de"}
FUNCTION {bbl.semicolon} { "; "}
FUNCTION {bbl.same.field} {"\underline{\ \ \ \ \ \ \ \ }"}
FUNCTION {bbl.techrep} { "Rel. Tec." }
FUNCTION {bbl.void} {""}
FUNCTION {bbl.volume} { "v." }


MACRO {jan} {"jan."}
MACRO {feb} {"fev."}
MACRO {mar} {"mar."}
MACRO {apr} {"abr."}
MACRO {may} {"maio"}
MACRO {jun} {"jun."}
MACRO {jul} {"jul."}
MACRO {aug} {"ago."}
MACRO {sep} {"set."}
MACRO {oct} {"out."}
MACRO {nov} {"nov."}
MACRO {dec} {"dez."}


FUNCTION {is.num}
{ chr.to.int$
  duplicate$ "0" chr.to.int$ < not
  swap$ "9" chr.to.int$ > not and
}
FUNCTION {str.to.int}
{ duplicate$ is.num
    {chr.to.int$ #48 -}
    'skip$
  if$
}
FUNCTION {bibinfo.check} %usage: field "field" bibinfo.check
{ swap$ duplicate$ missing$
    {pop$ pop$ ""}
    { duplicate$ empty$
        {swap$ pop$}
        {swap$ pop$}
      if$
    }
  if$
}
FUNCTION {bibinfo.warn}
{ swap$
  duplicate$ missing$
    {swap$ "faltando " swap$ * " em " * cite$ * warning$ pop$ ""}
    { duplicate$ empty$
        {swap$ "vazio " swap$ * " em " * cite$ * warning$}
        {swap$ pop$}
      if$
    }
  if$
}

FUNCTION {write.field.verbatim}
{
duplicate$ missing$
  {pop$ pop$} 
  {
  s text.length$ #0 >
    {s "," * newline$ write$ "" 's :=}
    'skip$
  if$
  swap$
  "={" * swap$ * "}" *
  s swap$ * 's :=
  }
if$
}

STRINGS  { bibinfo}
INTEGERS { nameptr namesleft numnames }

FUNCTION {get.person} {"{vv }{ll}{, jj}{, ff}" format.name$}
FUNCTION {get.last.name} {#1 "{ll}" format.name$}
FUNCTION {get.first.name} {#1 "{ff}" format.name$}
FUNCTION {get.von} {#1 "{vv}" format.name$}
FUNCTION {get.jr} {#1 "{jj}" format.name$}

FUNCTION {recompose.name}
  {get.person duplicate$ duplicate$ duplicate$
   get.von #1 "{ll }" format.name$ swap$
   get.last.name *
   swap$ #1 "{, ff}" format.name$ *
   swap$ get.von #1 "{ ff}" format.name$ *
   }


FUNCTION {format.single.name}
{
s nameptr get.person 'r :=
r get.last.name
   get.last.name
   r #1 "{ jj}" format.name$ * "u" change.case$
	  r #1 "{, f.}{ vv}" format.name$ *
}
FUNCTION {format.names}
{ 'bibinfo :=  duplicate$ empty$ 'skip$
  {'s :=
   "" 't :=
   #1 'nameptr :=  s num.names$ 'numnames :=
    numnames 'namesleft :=
    { namesleft #0 > }
    { format.single.name
      bibinfo bibinfo.check
      't :=
      nameptr #1 >
        {nameptr #1 #1 + = numnames usp.max.nomes > and
            {usp.max.nomes {"others" 't := #1 'namesleft :=} 'skip$ if$ }
            'skip$
          if$
          namesleft #1 >
            { "; " * t * }
            { numnames #2 >
                { "" * }
                'skip$
              if$
              s nameptr "{ll}" format.name$ duplicate$ "others" =
                { 't := }
                { pop$ }
              if$
              t "others" =
                { bbl.etal *}
                { "; " * t *}
              if$ }
          if$ }
        't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft := }
  while$
  } if$
}

%from plain.bst
FUNCTION {format.names.plain}
{ 's :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { s nameptr "{ff }{vv }{ll}{ jj}" format.name$ 't :=
      nameptr #1 >
	{ namesleft #1 >
	    { ", " * t * }
	    { numnames #20 >
		{ "," * }
		'skip$
	      if$
	      t "others" =
		{ bbl.etal * }
		{ bbl.and * t * }
	      if$
	    }
	  if$
	}
	't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}
FUNCTION {format.names.ed} {format.names}

% ENTRADAS: author
FUNCTION {format.authors}
{
author "author" format.names
}

% ENTRADAS: editor
FUNCTION {formatar.editor}
{
editor "editor" format.names "(Ed.)" *
}

% ENTRADAS: compilador
FUNCTION {formatar.compilador}
{
compilador "compilador" format.names "(Comp.)" *
}

% ENTRADAS: organization
FUNCTION {formatar.organization}
{
organization "organization" format.names "(Org.)" *
}

% ENTRADAS: author
FUNCTION {format.authors.plain}
{
author format.names.plain
}

% ENTRADAS: pages
FUNCTION {format.book.pages}
{ pages "pages" bibinfo.check  duplicate$ empty$
    'skip$
    {  "~" * bbl.pages * }
  if$
}

% ENTRADAS: dimensions
FUNCTION {format.dimensions}
{ dimensions empty$
    'skip$
    { dimensions output}
  if$
}

% ENTRADAS: doi
FUNCTION {format.doi}
{ doi empty$
    { "" }
    {new.block "\href{http://dx.doi.org/" doi * "}{DOI: " * doi * "}" *}
  if$
}

% ENTRADAS: note
FUNCTION {format.note}
{ note empty$
    { "" }
    { note #1 #1 substring$
      duplicate$ "{" =
        'skip$
        { output.state mid.sentence =
          { "l" }
          { "u" }
        if$
        change.case$ }
      if$
      note #2 global.max$ substring$ * "note" bibinfo.check }
  if$
}

FUNCTION {se.letra}
{ chr.to.int$ 'letra :=
	letra #96 > letra #123 < and letra #64 > letra #91 < and or
}

FUNCTION {se.numero}
{ chr.to.int$ 'letra :=
	letra #47 > letra #58 < and
}

FUNCTION {fim.de.comando}
{ 'caractere :=
	caractere se.letra caractere se.numero or caractere "+" = or caractere "-" = or caractere "_" = or caractere "\" = or not
}

FUNCTION {formatar.titulo.como.autor}
{ 's := %holds the title
  #1 'title.pos := %our position in the title
  #0 'first.word := %boolean, true if we already have the first word
  #1 'no.bracket := %boolean, false if we are inside a {}
  #1 'nao.comando := %boolean para verificar se trata-se de comando
  "" 't := %holds the first title.pos characters of the title
  {first.word #0 =}
  {s title.pos #1 substring$ 'title.char :=
  title.char "\" = {#0 'nao.comando :=} 'skip$ if$
  title.char se.letra {nao.comando {#1 'first.word := title.pos #1 - 'title.pos :=} 'skip$ if$} {title.char fim.de.comando {#1 'nao.comando :=} 'skip$ if$ } if$
  title.pos #1 + 'title.pos :=
  }
  while$
  s #1 title.pos #1 - substring$
  {first.word}
    {s title.pos #1 substring$ 'title.char :=
      title.char "{" = {#0 'no.bracket :=} 'skip$ if$
      no.bracket 'skip$ {title.char "}" = {#1 'no.bracket :=} 'skip$ if$}  if$
	  title.char " " = s t = or no.bracket and
        {#0 'first.word :=
        t title.char * "u" change.case$
        title.pos #1 + 'title.pos :=
        s title.pos global.max$ substring$ * 's :=}
        {title.pos #1 + 'title.pos := t title.char * 't :=}
     if$
     }
  while$
s *
}

FUNCTION {para.maiuscula}
{ 's := %holds the title
	#0 'protegido :=
	#0 'brackets.abertos :=
	#1 'title.pos :=
	#1 'nao.comando :=
	s text.length$ #1 + 'tamanho.string :=
	"" 't :=
	"" 'title.char :=
	{title.pos tamanho.string = not}
	{s title.pos #1 substring$ 'title.char :=
		"{" title.char = "}" title.char = or {tamanho.string #1 + 'tamanho.string :=} 'skip$ if$
		protegido 
		{"}" title.char =
			{brackets.abertos #1 - 'brackets.abertos :=}
			'skip$
			if$
			"{" title.char =
			{brackets.abertos #1 + 'brackets.abertos :=}
			'skip$
			if$
			brackets.abertos #0 =
			{#0 'protegido :=}
			'skip$
			if$
			protegido {t title.char * 't :=} 'skip$ if$} 
		{title.char se.letra nao.comando and
			{title.char "u" change.case$ 'title.char :=}
			'skip$
			if$
			title.char "{" = nao.comando and
			{#1 'protegido := #1 'brackets.abertos :=}
			{title.char "\" = {#0 'nao.comando :=} 'skip$ if$
				title.char fim.de.comando {#1 'nao.comando :=} 'skip$ if$}
			if$
			protegido 'skip$ {t title.char * 't :=} if$}
		if$
		title.pos #1 + 'title.pos :=}
	while$
	t
}

FUNCTION {para.minuscula}
{ 's := %holds the title
	#0 'protegido :=
	#0 'brackets.abertos :=
	#1 'title.pos :=
	#1 'nao.comando :=
	s text.length$ #1 + 'tamanho.string :=
	"" 't :=
	"" 'title.char :=
	{title.pos tamanho.string = not}
	{s title.pos #1 substring$ 'title.char :=
		"{" title.char = "}" title.char = or {tamanho.string #1 + 'tamanho.string :=} 'skip$ if$
		protegido 
		{"}" title.char =
			{brackets.abertos #1 - 'brackets.abertos :=}
			'skip$
			if$
			"{" title.char =
			{brackets.abertos #1 + 'brackets.abertos :=}
			'skip$
			if$
			brackets.abertos #0 =
			{#0 'protegido :=}
			'skip$
			if$
			protegido {t title.char * 't :=} 'skip$ if$} 
		{title.char se.letra nao.comando and
			{title.char "l" change.case$ 'title.char :=}
			'skip$
			if$
			title.char "{" = nao.comando and
			{#1 'protegido := #1 'brackets.abertos :=}
			{title.char "\" = {#0 'nao.comando :=} 'skip$ if$
				title.char fim.de.comando {#1 'nao.comando :=} 'skip$ if$}
			if$
			protegido 'skip$ {t title.char * 't :=} if$}
		if$
		title.pos #1 + 'title.pos :=}
	while$
	t
}

FUNCTION {para.primeira.maiuscula}
{ 's := %holds the title
	#0 'protegido :=
	#0 'brackets.abertos :=
	#1 'title.pos :=
	#1 'nao.comando :=
	#1 'primeira.letra :=
	s text.length$ #1 + 'tamanho.string :=
	"" 't :=
	"" 'title.char :=
	{title.pos tamanho.string = not}
	{s title.pos #1 substring$ 'title.char :=
		"{" title.char = "}" title.char = or {tamanho.string #1 + 'tamanho.string :=} 'skip$ if$
		protegido 
		{ %{
			"}" title.char =
			{brackets.abertos #1 - 'brackets.abertos :=}
			'skip$
			if$
			"{" title.char = %}
			{brackets.abertos #1 + 'brackets.abertos :=}
			'skip$
			if$
			brackets.abertos #0 =
			{#0 'protegido :=}
			'skip$
			if$
			title.char se.letra title.char se.numero or {#0 'primeira.letra :=} 'skip$ if$
			protegido {t title.char * 't :=} 'skip$ if$} 
		{title.char se.letra nao.comando and
			{primeira.letra {title.char "u" change.case$ 'title.char :=} {title.char "l" change.case$ 'title.char :=} if$}
			'skip$
			if$
			title.char "{" = nao.comando and %}
			{#1 'protegido := #1 'brackets.abertos :=}
			{title.char "\" = {#0 'nao.comando :=} 'skip$ if$
				title.char fim.de.comando {#1 'nao.comando :=} 'skip$ if$}
			if$
			title.char se.letra title.char se.numero or nao.comando and {#0 'primeira.letra :=} 'skip$ if$
			protegido 'skip$ {t title.char * 't :=} if$}
		if$
		title.pos #1 + 'title.pos :=}
	while$
	t
}

FUNCTION {format.title.as.author}
{ 's := %holds the title
  #1 'title.pos := %our position in the title
  #1 'first.word := %boolean, true if we already have the first word
  #1 'no.bracket := %boolean, false if we are inside a {}
  "" 't := %holds the first title.pos characters of the title
  {first.word}
    {s title.pos #1 substring$ 'title.char :=
      %checks if the character is a {
      title.char "{" = {#0 'no.bracket :=} 'skip$ if$
      %if we are in a bracket
      no.bracket 'skip$ {title.char "}" = {#1 'no.bracket :=} 'skip$ if$}  if$
      type$ "journalpart" =
        {title.char "." = title.char ":" = or title.char "!" = or
	 title.char "," = or title.char ";" = or title.char "?" = or}
	{title.char " " =}
      if$ s t = or no.bracket and
       {#0 'first.word :=
       t title.char * "u" change.case$
       title.pos #1 + 'title.pos :=
       s title.pos global.max$ substring$ * 's :=}
       {title.pos #1 + 'title.pos := t title.char * 't :=}
     if$
     }
  while$
s
}

% ENTRADAS: title,subtitle
FUNCTION {format.title}
{author empty$ editor empty$ and organization empty$ and compilador empty$ and jurisdiction empty$ and entidade empty$ and
  {title formatar.titulo.como.autor}
  {title  duplicate$ empty$ {pop$ ""} {type$ "juridical" = 'skip$ {para.primeira.maiuscula} if$} if$}
 if$
 subtitle  empty$ 'skip$ {subtitle ": " swap$ * para.primeira.maiuscula *} if$
}

FUNCTION {n.dashify}
{
  't :=
  ""
    { t empty$ not }
    { t #1 #1 substring$ "-" =
        { t #1 #2 substring$ "--" = not
            { "--" *
              t #2 global.max$ substring$ 't :=
            }
            {   { t #1 #1 substring$ "-" = }
                { "-" *
                  t #2 global.max$ substring$ 't :=
                }
              while$
            }
          if$
        }
        { t #1 #1 substring$ *
          t #2 global.max$ substring$ 't :=
        }
      if$
    }
  while$
}

FUNCTION {word.in} { bbl.in " " * }

% ENTRADAS: year,month
FUNCTION {format.date}
{ year empty$
    { month empty$
	{ bbl.missing.year }
	{ "existe o campo month (mes) mas nao o campo year (ano) em " cite$ * warning$ month}
      if$}
    { month empty$
	{ year n.dashify}
	{ month " " * year * }
      if$}
  if$
}

% ENTRADAS: year,month,day
FUNCTION {format.full.date}
{ month empty$
    {#0}
	{month #1 #1 substring$ chr.to.int$ duplicate$ #64 > swap$ #91 < and {#1} {#0} if$}
  if$
  'month.first := %Se o mes iniciar com letra maiuscula significa que e idioma alemao ou ingles e o mes deve aparecer antes do dia
  day empty$
		{month empty$
			{year empty$
				'skip$
				{year output}
			if$}
			{month year empty$
				'skip$
				{" " * year *}
			if$ output}
		if$}
		{month.first {""} {day} if$ month empty$
			{year empty$
				'skip$
				{" " * year *}
			if$}
			{" " * month * month.first {" " * day *} {} if$ year empty$
				'skip$
				{" " * year *}
			if$}
		if$ output}
	if$
}

% ENTRADAS: title,subtitle
FUNCTION {format.btitle}
{author empty$ editor empty$ and organization empty$ and compilador empty$ and entidade empty$ and
    {title formatar.titulo.como.autor "title" bibinfo.check duplicate$ empty$ 'skip$ {} if$
    subtitle  empty$ 'skip$ {subtitle ": " swap$ * para.primeira.maiuscula *} if$}
    {title "title" bibinfo.check emphasize duplicate$ empty$ 'skip$ {} if$
    subtitle  empty$ 'skip$ {subtitle ": " swap$ * para.primeira.maiuscula *} if$}
if$
}

% ENTRADAS: tradutor
FUNCTION {format.tradutor}
{ tradutor empty$ 'skip$ {" Tradu\c{c}\~ao de " tradutor * "." * output new.block} if$}

% ENTRADAS: revisor
FUNCTION {format.revisor}
{ revisor empty$ 'skip$ {" Revis\~ao de " revisor * "." * output new.block} if$}

% ENTRADAS: ilustrador
FUNCTION {format.ilustrador}
{ ilustrador empty$ 'skip$ {" Ilustra\c{c}\~ao de " ilustrador * "." * output} if$}

% ENTRADAS: direcao
FUNCTION {format.direcao}
{ direcao empty$ 'skip$ {" Dire\c{c}\~ao geral de " direcao * "." * output new.block} if$}

FUNCTION {either.or.check}
{ empty$
    'pop$
    { "nao posso usar ambos " swap$ * " os campos em " * cite$ * warning$ }
  if$
}

% ENTRADAS: volume
FUNCTION {format.volume} %leaves formatted or empty volume on the stack
{volume duplicate$ empty$
   'skip$
   {volume #1 #1 substring$ is.num
      {bbl.volume swap$ tie.or.space.prefix * *}
      'skip$
    if$}
 if$
}

% ENTRADAS: volumes
FUNCTION {format.volumes}
{volumes empty$
   'skip$
   { volumes bbl.volume tie.or.space.prefix * * *}
 if$
}

% ENTRADAS: number
FUNCTION {format.number} %leaves formatted or empty number on the stack
{ number duplicate$ empty$
    'skip$
    {bbl.number swap$ tie.or.space.prefix * *}
  if$
}

% ENTRADAS: number
FUNCTION {format.patent.number}
{ number empty$
    'skip$
	{number output}
  if$
}

% ENTRADAS: volume,number,series
FUNCTION {format.bvolume}
{ volume empty$
    { "" }
    { format.volume
      number empty$ 'skip$ {output format.number} if$
      series "series" bibinfo.check
      duplicate$ empty$ 'pop$
        { swap$ bbl.of space.word * swap$
          emphasize * }
      if$
    }
  if$
}

% ENTRADAS: volume,number,series
FUNCTION {format.bvolume.if.no.series}
{ series empty$
    { format.bvolume }
    { "" }
  if$
}

% ENTRADAS: number,volume
FUNCTION {format.number.or.volume}
{number empty$
  {format.volume}
  {number} %we use the pure number here without leading "n."
 if$
}

% ENTRADAS: volume,number
FUNCTION {format.volume.and.number}
{format.volume output
 format.number
}

% ENTRADAS: number,volume,series,volumes
FUNCTION {format.volume.number.series}
{ number empty$ volume empty$ and
  { series empty$
    { "" }
    {" (" series "series" bibinfo.check * ")" *}
    if$}
  { series empty$
    { format.number.or.volume }
    { " ("  series "series" bibinfo.check * ", " *
     format.number.or.volume * ")" *  }
     if$ }
  if$
  format.volumes
}

% ENTRADAS: number,volume,series
FUNCTION {format.bvolume.number.series}
{ number empty$ volume empty$ and
  { series empty$
    { "" }
    {" (" series "series" bibinfo.check * ")" *}
    if$}
  { series empty$
    { format.volume.and.number }
    { " ("  series "series" bibinfo.check * ", " *
     format.volume.and.number * ")" *  }
     if$ }
  if$
}

FUNCTION {extract.num}
{ duplicate$ 't :=
  "" 's :=
  { t empty$ not }
  { t #1 #1 substring$
    t #2 global.max$ substring$ 't :=
    duplicate$ is.num
      { s swap$ * 's := }
      { pop$ "" 't := }
    if$ }
  while$
  s empty$
    'skip$
    { pop$ s }
  if$
}

% ENTRADAS: edition
FUNCTION {format.edition}
{ edition duplicate$ empty$ 'skip$
    {output.state mid.sentence = { "l" } { "t" } if$ change.case$
      "edition" bibinfo.check
    edition #1 #1 substring$ is.num {". " * bbl.edition * } 'skip$ if$}
  if$
}
INTEGERS { multiresult }
FUNCTION {multi.page.check}
{ 't :=
  #0 'multiresult :=
    { multiresult not
      t empty$ not
      and }
    { t #1 #1 substring$
      duplicate$ "-" =
      swap$ duplicate$ "," =
      swap$ "+" =
      or or
        { #1 'multiresult := }
        { t #2 global.max$ substring$ 't := }
      if$ }
  while$
  multiresult
}

% ENTRADAS: pages
FUNCTION {format.pages}
{ pages duplicate$ empty$ 'skip$
    { duplicate$ multi.page.check
        { n.dashify bbl.pages swap$}
        { bbl.page swap$ }
      if$
      tie.or.space.prefix "pages" bibinfo.check * * }
  if$
}

% ENTRADAS: numpages
FUNCTION {format.numpages}
{ numpages duplicate$ empty$ 'skip$
    { pop$ numpages bbl.pages tie.or.space.prefix * * }
  if$
}

% ENTRADAS: conference-number,conference-year,conference-location
FUNCTION {format.conference.info}
{ conference-number empty$
    'skip$
    {conference-number "." * output}
  if$
  conference-year empty$
    'skip$
    {conference-year output}
  if$
  conference-location empty$
    'skip$
    {conference-location output}
  if$
  new.sentence
}

% ENTRADAS: pages,chapter,type
FUNCTION {format.chapter.pages}
{ chapter empty$
    'format.pages
    { type empty$
        { bbl.chapter }
        { type para.minuscula
          "type" bibinfo.check }
      if$
      chapter tie.or.space.prefix "chapter" bibinfo.check * *
      pages empty$
        'skip$
        { ", " * format.pages * }
      if$ }
  if$
}

% ENTRADAS: pages,chapter
FUNCTION {format.bchapter.pages}
{ chapter empty$
    'skip$
    {chapter output.nonnull}
  if$
  pages empty$
    'skip$
	{"p. " pages * output.nonnull}
  if$
}

% ENTRADAS: booktitle,booksubtitle
FUNCTION {format.booktitle}
{author empty$ editor empty$ and organization empty$ and jurisdiction empty$ and entidade empty$ and compilador empty$ and evento empty$ and
    {booktitle formatar.titulo.como.autor "booktitle" bibinfo.check duplicate$ empty$ 'skip$ {} if$
    booksubtitle  empty$ 'skip$ {booksubtitle ": " swap$ * para.primeira.maiuscula *
	} if$}
    {booktitle "booktitle" bibinfo.check emphasize duplicate$ empty$ 'skip$ {} if$
    booksubtitle  empty$ 'skip$ {booksubtitle ": " swap$ * para.primeira.maiuscula *
	} if$}
if$
}

% ENTRADAS: editor
FUNCTION {format.editors}
{ editor  empty$
   {"inbook" type$ = 'bbl.same.field 'skip$ if$}
   {author empty$
    {'bbl.same.field }
    {"inbook" type$ =
      {editor "editor" format.names }
      {editor "editor" format.names " (" *
       bbl.editor
       ")" * *}
    if$}
   if$}
  if$
}

% ENTRADAS: booktitle,booksubtitle,inauthor,ineditor,inorganization,incompilador,inentidade,inentidade-sub
FUNCTION {format.in.ed.booktitle}
{	word.in
	inauthor empty$
		{ ineditor empty$
			{ inorganization empty$
				{ incompilador empty$
					{ inentidade empty$
						{"inbook" type$ = {booktitle formatar.titulo.como.autor * booksubtitle empty$ 'skip$ {": " * booksubtitle *} if$}{booktitle empty$ 'skip$ {format.booktitle *} if$} if$}
						{ inentidade para.maiuscula * inentidade-sub empty$ 'skip$ {": " * inentidade-sub *} if$}
					if$}
					{ incompilador "incompilador" format.names * " (Comp.)" *}
				if$}
				{ inorganization "inorganization" format.names * " (Org.)" *}
			if$}
			{ ineditor "ineditor" format.names * " (Ed.)" *}
		if$}
		{ inauthor "inauthor" format.names *}
	if$
	output new.block
	inauthor empty$ ineditor empty$ and inorganization empty$ and incompilador empty$ and inentidade empty$ and "inbook" type$ = booktitle empty$ or and
		'skip$
		{format.booktitle output new.block}
	if$
}

% ENTRADAS: booktitle,booksubtitle
FUNCTION {format.in.booktitle}
{
	booktitle empty$ 'skip$
		{word.in booktitle emphasize * booksubtitle empty$ 'skip$ {": " * booksubtitle *} if$}
	if$
	output
	new.sentence
}

% ENTRADAS: evento,conference-number,conference-year,conference-location
FUNCTION {format.in.conference}{
	word.in evento empty$ 'skip$ {evento para.maiuscula *} if$ output
	format.conference.info
}

% ENTRADAS: type
FUNCTION {format.type}
{ type "type" bibinfo.check
  duplicate$ empty$ 'skip$
    {new.block "" swap$ * }
  if$
}

% ENTRADAS: year
FUNCTION {format.year}
{year empty$
   {bbl.missing.year}
    {year n.dashify 
     "year" bibinfo.check}
if$
}

% ENTRADAS: year-presented
FUNCTION {format.year.presented}
{year-presented empty$
   {""}
   {year-presented n.dashify}
if$
}

% ENTRADAS: address
FUNCTION {format.field.address}
{ 't :=
  entidade missing$ {""} {entidade empty$ {""} {entidade} if$} if$ 's :=
  journal missing$ {""} {journal empty$ {""} {journal} if$} if$ 'r :=
  address empty$ organization empty$ and t empty$ and
    {bbl.no.address.no.publisher}
    {address empty$
      {bbl.no.address}
      {address}
    if$
    t empty$
      {%if an organization is present then it is OK not to have a publisher
       organization empty$ {": " * bbl.no.publisher * } 'skip$ if$}
      {s t = not r t = not and {": " * t *} 'skip$ if$}
     if$}
  if$
}

% ENTRADAS: address
FUNCTION {format.address}
{address empty$
  { bbl.no.address}
  { address}
 if$
}

% ENTRADAS: publisher,address
FUNCTION {format.publisher.address}
{ publisher "publisher" bibinfo.check format.field.address}

% ENTRADAS: publisher,address
FUNCTION {format.publisher.address.relax}
{publisher empty$ address empty$ and {""}
 {publisher "publisher" bibinfo.check format.field.address}
if$}

% ENTRADAS: organization,address
FUNCTION {format.organization.address}
{ organization "organization" bibinfo.check format.field.address}

% ENTRADAS: author,organization
FUNCTION {format.author.or.organization}
{
  author empty$
    { organization empty$ 'skip$
      { organization para.maiuscula "organization"  bibinfo.check output}
      if$}
    { format.authors output.nonnull }
  if$
}

% ENTRADAS: entidade,entidade-sub
FUNCTION {formatar.entidade}{
	entidade empty$ 
		'skip$
		{ entidade para.maiuscula entidade-sub empty$ 'skip$ {". " * entidade-sub *} if$ output.nonnull}
	if$
}

% ENTRADAS: author,editor,compilador,organization,entidade,entidade-sub
FUNCTION {formatar.autor.ou.organizador.ou.editor.ou.compilador}
	{  author empty$
		{ editor empty$
			{ compilador empty$
				{ organization empty$
					{ entidade empty$ 'skip$
						{ entidade para.maiuscula entidade-sub empty$ 'skip$ {". " * entidade-sub *} if$ output.nonnull}
					if$}
					{ formatar.organization output.nonnull }
				if$}
				{ formatar.compilador output.nonnull }
			if$}
			{ formatar.editor output.nonnull }
		if$}
		{ format.authors output.nonnull }
	if$
}

% ENTRADAS: author,entidade,entidade-sub
FUNCTION {formatar.entidade.e.ou.autor}
{	entidade empty$
		{author empty$
			'skip$
			{format.authors.plain para.maiuscula output}
		if$}
		{entidade para.maiuscula entidade-sub empty$ 'skip$ {". " * entidade-sub *} if$
		author empty$
			'skip$
			{". " * format.authors.plain * output}
		if$}
	if$
}

% ENTRADAS: jurisdiction
FUNCTION {formatar.jurisdicao}
{
	jurisdiction empty$ 'skip$
		{ jurisdiction para.maiuscula "jurisdiction"  bibinfo.check output }
	if$
}

% ENTRADAS: editor,organization
FUNCTION {format.editor.or.organization}
{  editor empty$
    { organization empty$ 'skip$
      { organization para.maiuscula "organization"  bibinfo.check output}
      if$}
    { format.editors output.nonnull }
  if$
}

% ENTRADAS: author,editor,organization
FUNCTION {format.author.or.editor.or.organization}
{ author empty$
    { editor empty$
        { organization empty$
	    'skip$
            { organization para.maiuscula "organization"  bibinfo.check output}
          if$}
        { format.editors output.nonnull }
      if$}
    { format.authors output.nonnull }
  if$
}

% ENTRADAS: journal,journal-sub
FUNCTION {format.journal}
{
  journal emphasize duplicate$ "" = journal-sub empty$ or 'skip$ {": " * journal-sub *} if$
}

% ENTRADAS: depoimento
FUNCTION {format.depoimento}
{
	depoimento empty$ 'skip$ {"[" depoimento * "]" * output.nonnull} if$
}

FUNCTION {format.names.for.label1}
{ 's := "" 't :=
   #1 'nameptr :=  s num.names$ 'numnames :=
    numnames 'namesleft :=
    { namesleft #0 > }
    {s nameptr "{ll}" format.name$ get.last.name
     s nameptr "{ jj}" format.name$ *
     't :=
      nameptr #1 >
        {namesleft #1 >
          { numnames usp.max.nomes > usp.max.nomes #0 = not and
	    {#0 'namesleft := usp.max.nomes {bbl.etal *} 'skip$ if$}
	    {", " * t * } if$ }
            { t "others" =
              {usp.max.nomes {bbl.etal *} 'skip$ if$}
                {numnames nameptr =
                {bbl.and * t *}
		'skip$ if$}
             if$
	     }
          if$ }
        't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft := }
  while$
}

FUNCTION {format.names.for.label2}
{ 's := "" 't :=
   #1 'nameptr :=  s num.names$ 'numnames :=
    numnames 'namesleft :=
    { namesleft #0 > }
    {s nameptr "{ll}" format.name$ get.last.name
     s nameptr "{ jj}" format.name$ *
 	 "u" change.case$ 't :=
      nameptr #1 >
        {namesleft #1 >
            { numnames usp.max.nomes > 
	      {#0 'namesleft := usp.max.nomes {bbl.etal *} 'skip$ if$} 
	      {"; " * t * } if$ }
            { t "OTHERS" = t "others" = or
	      {usp.max.nomes {bbl.etal *} 'skip$ if$}
	      {numnames nameptr = { bbl.semicolon * t *} 'skip$ if$}
             if$
	     }
          if$ }
        't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft := }
  while$
}

FUNCTION {format.title.for.label}
{  's :=
  #1 'title.pos :=
  #0 'first.word :=
  #1 'no.bracket := %boolean, false if we are inside a {}
  #1 'nao.comando := %boolean para verificar se trata-se de comando
  {first.word #0 =}
  {s title.pos #1 substring$ 'title.char :=
  title.char "\" = {#0 'nao.comando :=} 'skip$ if$
  title.char se.letra {nao.comando {#1 'first.word := title.pos #1 - 'title.pos :=} 'skip$ if$} {title.char fim.de.comando {#1 'nao.comando :=} 'skip$ if$ } if$
  title.pos #1 + 'title.pos :=
  }
  while$
  title.pos #1 = {""} {s #1 title.pos #1 - substring$} if$ 't :=
  {first.word}
    {s title.pos #1 substring$ 'title.char :=
        title.char "." = title.char ":" = or title.char ";" = or
	title.char "," = or title.char " " = or s t = or
       {#0 'first.word :=
       title.pos #1 + 'title.pos :=}
       {title.pos #1 + 'title.pos := t title.char * 't :=}
     if$
     }
  while$
s t = %title equals first word, dont use bbl.ldots
  {t  "}{" * t para.maiuscula *}
  {t bbl.ldots * "}{" * t para.maiuscula * bbl.ldots *}
if$
}

%cvs keys have the general form $key: value $
INTEGERS {key.pos value.pos}
FUNCTION {extract.cvs.key} %we borrow the functionality of format.title.for.label
{ 's := %holds the entire cvs string
  #1 'key.pos :=
  #0 'value.pos := %0= between first $ and :, 1=between : and second $, 2= at the end
  "" 't := %holds the resulting value
  {value.pos #0 = value.pos #1 = or}
    { s key.pos #1 substring$ 'title.char :=
      value.pos #0 =
      { title.char ":" = {#1 'value.pos := key.pos #1 + 'key.pos :=} 'skip$ if$}
      { title.char "$" =
        {#2 'value.pos :=}
        {t title.char * 't :=}
	if$}
     if$
     key.pos #1 + 'key.pos :=
     }
  while$
  t
}

FUNCTION {pega.autor.editor.comp.titulo}
{	author empty$ not
		{author format.names.for.label1 * "}{" * author format.names.for.label2 *}
		{editor empty$ not
			{editor format.names.for.label1 * "}{" *  editor format.names.for.label2 *}
			{compilador empty$ not
				{compilador format.names.for.label1 * "}{" * compilador format.names.for.label2 *}
				{organization empty$ not
					{organization format.names.for.label1 * "}{" * organization format.names.for.label2 *}
					{title empty$ not
						{title format.title.for.label *}
						{"nada}{NADA" *}
					if$}
				if$}
			if$}
		if$}
	if$
}

FUNCTION {periodical.label}
{	journal empty$ not
		{journal * "}{" * journal para.maiuscula *}
		{"nada}{NADA}{" *}
	if$
}

FUNCTION {conference.label}
{	evento empty$ not
		{evento * "}{" * evento para.maiuscula *}
		{"nada}{NADA}{" *}
	if$
}

FUNCTION {patent.label}
{
	entidade empty$ not
		{entidade * "}{" * entidade para.maiuscula *}
		{editor empty$ not
			{editor format.names.for.label1 * "}{" * editor format.names.for.label2 *}
			{organization empty$ not
				{organization format.names.for.label1 * "}{" *  organization format.names.for.label2 *}
				{author empty$ not
					{author format.names.for.label1 * "}{" * author format.names.for.label2 *}
					{title empty$ not
						{title format.title.for.label *}
						{"nada}{NADA" *}
					if$}
				if$}				
			if$}
		if$}
	if$
}

FUNCTION {juridical.label}
{	jurisdiction empty$ not
		{jurisdiction * "}{" * jurisdiction para.maiuscula *}
		{title empty$ not
			{title format.title.for.label *}
			{"nada}{NADA}{" *}
		if$}
	if$
}

FUNCTION {formata.forma.sentenca}
{	's :=
	#0 'posicao.substituicao :=
	#1 'title.pos :=
	s text.length$ #1 + 'tamanho.string :=
	{title.pos tamanho.string = not}
		{s title.pos #1 substring$ 'title.char :=
		title.char ";" = {title.pos 'posicao.substituicao :=} 'skip$ if$
		title.char "{" = title.char "}" = or {tamanho.string #1 + 'tamanho.string :=} 'skip$ if$ title.pos #1 + 'title.pos :=}
	while$
	posicao.substituicao #0 = {s} {s #1 posicao.substituicao #1 - substring$ " e " * s posicao.substituicao #1 + global.max$ substring$ *} if$
}

FUNCTION {other.label}
{	forma empty$ not
		{forma formata.forma.sentenca * "}{" * forma para.maiuscula *}
		{"nada}{NADA" *}
	if$
}

FUNCTION {article.label}
{	author empty$ not
		{author format.names.for.label1 * "}{" * author format.names.for.label2 *}
		{editor empty$ not
			{editor format.names.for.label1 * "}{" *  editor format.names.for.label2 *}
			{compilador empty$ not
				{compilador format.names.for.label1 * "}{" * compilador format.names.for.label2 *}
				{organization empty$ not
					{organization format.names.for.label1 * "}{" * organization format.names.for.label2 *}
					{entidade empty$ not
						{entidade * "}{" * entidade para.maiuscula *}
						{title empty$ not
							{title format.title.for.label *}
							{"nada}{NADA" *}
						if$}
					if$}
				if$}
			if$}
		if$}
	if$
}

FUNCTION {calc.extended.label}
{ "{"
	#0 'escolhido :=
	type$ "periodical" =
		{ periodical.label #1 'escolhido :=}
		'skip$
	if$
	type$ "conference" =
		{conference.label #1 'escolhido :=}
		'skip$
	if$
	type$ "patent" =
		{patent.label #1 'escolhido :=}
		'skip$
	if$
	type$ "juridical" =
		{juridical.label #1 'escolhido :=}
		'skip$
	if$
	type$ "other" =
		{other.label #1 'escolhido :=}
		'skip$
	if$
	escolhido not
		{article.label}
		'skip$
	if$
	"}{" *
	year empty$ 'skip$ {year *} if$
	'label := %place the stack into label
 % final } will be placed in reverse.pass
}
%generates and simple label of type "Author YEAR"
%final result is placed onto the stack

FUNCTION {calc.simple.label}
{ author empty$ not
    {author format.names.for.label1}
    {editor empty$ not
     {editor format.names.for.label1}
     {organization empty$ not
      {organization}
      {title empty$ not
       {title}
       {""}
      if$}
     if$}
    if$}
   if$
 year empty$ 'skip$ {" " * year *} if$
}

FUNCTION {output.bibitem}
{ newline$
  "\adicionaformascitacoes{" write$ cite$ write$ "}" write$ newline$
  label write$ newline$
  "\uspbibitem{" write$ cite$ write$ "}" write$ newline$
  "{" write$
  "" before.all 'output.state :=
}
FUNCTION {fin.entry}
{ add.period$ write$
  "}" write$
  newline$
}

% ENTRADAS: local-publicaco,author,editor,compilador,organization,entidade,entidade-sub,publisher,address,type,direcao,ilustrador,revisor,tradutor,title, subtitle,year,month,dimensions,pages,edition,number,volume,series,volumes,doi,note
FUNCTION {book}
{ output.bibitem
  formatar.autor.ou.organizador.ou.editor.ou.compilador new.sentence
  format.type output new.sentence
  format.btitle "title" output.check new.sentence
  format.tradutor
  format.revisor
  format.direcao
  format.ilustrador new.sentence
  format.edition output new.sentence
  format.publisher.address output
  local-publicacao empty$ 'skip$ {local-publicacao output new.block} if$
  format.date output new.sentence
  format.book.pages output
  format.dimensions  new.sentence
  format.volume.number.series output new.sentence
  format.doi output new.block
  format.note output new.block
  fin.entry
}

% ENTRADAS: local-publicaco,author,editor,compilador,organization,entidade,entidade-sub,publisher,address,type,direcao,ilustrador,revisor,tradutor,title, subtitle,year,month,dimensions,pages,edition,number,volume,series,volumes,doi,note
FUNCTION {manual} {book}

% ENTRADAS: local-publicaco,author,editor,compilador,organization,entidade,entidade-sub,publisher,address,type,direcao,ilustrador,revisor,tradutor,title, subtitle,year,month,dimensions,pages,edition,number,volume,series,volumes,doi,note
FUNCTION {standard} {book}

% ENTRADAS: local-publicaco,author,editor,compilador,organization,entidade,entidade-sub,publisher,address,type,direcao,ilustrador,revisor,tradutor,title, subtitle,year,month,dimensions,pages,edition,number,volume,series,volumes,doi,note
FUNCTION {techreport} {book}

% ENTRADAS: address,entidade,entidade-sub,title,subtitle,year,month,pages,number,volume,series,volumes,doi,note
FUNCTION {booklet}
{ output.bibitem
  formatar.entidade new.sentence
  format.btitle "title" output.check new.sentence
  address "address" bibinfo.check output
  format.date output new.sentence
  format.book.pages output
  format.volume.number.series output new.sentence
  format.doi output new.block
  format.note output new.block
  fin.entry
}

% ENTRADAS: address,author,editor,compilador,organization,entidade,entidade-sub,title,subtitle,year,month,pages,edition,number,volume,series,volumes,doi,note
FUNCTION {laudo} 
{ output.bibitem
  formatar.autor.ou.organizador.ou.editor.ou.compilador new.sentence
  format.btitle "title" output.check new.sentence
  format.edition output new.sentence
  address "address" bibinfo.check output
  format.date output new.sentence
  format.book.pages output
  format.volume.number.series output new.sentence
  format.doi output new.block
  format.note output new.block
  fin.entry
}

% ENTRADAS: genero,type
FUNCTION {format.thesis.type}
{ genero empty$
    {type$ "phdthesis" =
      {bbl.phdthesis}
      {type$ "mastersthesis" =
         {bbl.mthesis}
         {type$ "monography" =
            {bbl.monography}
	        {bbl.void}
         if$}
      if$}
    if$}
    {genero}
  if$
  type empty$
    {type missing$
       {type$ "phdthesis" =
          { " (" bbl.phd * ")" *}
          {type$ "mastersthesis" = { " (" bbl.master * ")" *} {pop$ ""} if$}
        if$}
       {""}
     if$}
     {type$ "thesis" =
       {type}
       {" (" type  * ")" * "type" bibinfo.check}
      if$}
  if$
 * %joins first and second part of this function
}

% ENTRADAS: school,address,genero,type,year-presented,year,title,subtitle,pages,author,volume,doi,note
FUNCTION {thesis}
{ output.bibitem
  format.authors "author" output.check new.block
  format.btitle "title" output.check new.block
  format.year output new.sentence
  format.book.pages output new.sentence
  format.volume output new.sentence
  format.thesis.type output.nonnull
  school "school" bibinfo.warn output.dash
  address "address" bibinfo.check output
  format.year.presented output new.sentence
  format.doi output new.block
  format.note output new.block
  fin.entry
}

% ENTRADAS: school,address,genero,type,year-presented,year,title,subtitle,pages,author,volume,doi,note
FUNCTION {phdthesis}     { thesis }

% ENTRADAS: school,address,genero,type,year-presented,year,title,subtitle,pages,author,volume,doi,note
FUNCTION {mastersthesis} { thesis }

% ENTRADAS: school,address,genero,type,year-presented,year,title,subtitle,pages,author,volume,doi,note
FUNCTION {monography}    { thesis }

% ENTRADAS: author,editor,compilador,organization,entidade,entidade-sub,publisher,address,booktitle,booksubtitle,inauthor,ineditor,inorganization, incompilador,inentidade,inentidade-sub,year,month,title,subtitle,dimensions,pages,chapter,type,edition,number,volume,series,volumes,doi,note
FUNCTION {inbook}
{ output.bibitem
  formatar.autor.ou.organizador.ou.editor.ou.compilador new.sentence
  format.title "title" output.check new.block
  format.in.ed.booktitle
  format.edition output new.sentence
  format.publisher.address output
  format.date output new.sentence
  format.volume.number.series output
  format.chapter.pages output new.sentence
  format.dimensions  new.sentence
  format.doi output new.block
  format.note output new.block
  fin.entry
}

% ENTRADAS: author,editor,compilador,organization,entidade,entidade-sub,publisher,address,booktitle,booksubtitle,inauthor,ineditor,inorganization, incompilador,inentidade,inentidade-sub,year,month,title,subtitle,dimensions,pages,chapter,type,edition,number,volume,series,volumes,doi,note
FUNCTION {incollection} {inbook}

% ENTRADAS: evento,publisher,address,booktitle,booksubtitle,conference-number,conference-year,conference-location,numpages,year,month,pages,doi,note
FUNCTION {conference}
{	output.bibitem
	evento empty$ 'skip$ {evento para.maiuscula output new.block} if$
	format.conference.info
	format.booktitle output new.block
	format.publisher.address output
	format.date output
	format.pages output new.block
	format.numpages output new.block
	format.doi output new.block
	format.note output new.block
	fin.entry
}

% ENTRADAS: author,editor,compilador,organization,entidade,entidade-sub,publisher,address,evento,conference-number,conference-year,conference-location, booktitle,booksubtitle,year,month,title,subtitle,pages,number,volume,series,volumes,doi,note
FUNCTION {inconference}
{	output.bibitem
	formatar.autor.ou.organizador.ou.editor.ou.compilador  new.sentence
	format.title "title" output.check new.block
	format.in.conference
	format.booktitle output new.block
	format.publisher.address output
	format.date output
	format.pages output new.block
	format.volume.number.series output
	format.doi output new.block
	format.note output new.block
	fin.entry
}

% ENTRADAS: editor,organization,publisher,address,conference-number,conference-year,conference-location,title,subtitle,year,month,pages,volume,number,series, volumes,doi,note
FUNCTION {proceedings}
{ output.bibitem
  format.editor.or.organization
  format.conference.info
  format.btitle "title" output.check
  format.bvolume output
  format.volume.number.series output
  editor empty$
    { publisher empty$
        'skip$
        { new.sentence format.publisher.address output format.date output}
      if$
    }
    { publisher empty$
        { new.sentence
          format.organization.address output format.date output}
        { new.sentence
          organization "organization" bibinfo.check output
          format.publisher.address output format.date output}
      if$ }
  if$
  new.sentence format.book.pages output new.sentence
  format.doi output new.block
  format.note output  new.block
  fin.entry
}

% ENTRADAS: evento,conference-number,conference-location,conference-year,author,editor,compilador,organization,entidade,entidade-sub,booktitle,booksubtitle, year,month,title,subtitle,pages,volume,number,series,doi,note
FUNCTION {inproceedings}
{	output.bibitem
	formatar.autor.ou.organizador.ou.editor.ou.compilador  new.sentence
	format.title "title" output.check new.block
	format.booktitle output new.block
	format.bvolume.if.no.series output
	format.pages output
	format.date output new.block
	evento empty$ 'skip$ {"Apresentado a " conference-number empty$ 'skip$ {conference-number * ". " *} if$ evento * output conference-location empty$ 'skip$ {conference-location output} if$ conference-year empty$ 'skip$ {conference-year output} if$ new.sentence} if$
	format.doi output new.block
	format.note output new.block
	fin.entry
}

% ENTRADAS: journal,periodo,publisher,address,year,month,dimensions,number,volume,series,doi,note
FUNCTION {periodical}
{ 	output.bibitem
	journal para.maiuscula output new.block
	format.publisher.address output
	format.bvolume.number.series output
	format.date output
	periodo "periodo" bibinfo.check output new.block
	format.dimensions  new.sentence
	format.doi output new.block
	format.note output new.block
	fin.entry
}

% ENTRADAS: address,publisher,depoimento,journal,journal-sub,author,editor,compilador,organization,entidade,entidade-sub,numpages,year,month,title,subtitle, pages,volume,number,doi,note
FUNCTION {article}
{ output.bibitem
  formatar.autor.ou.organizador.ou.editor.ou.compilador new.sentence
  format.title "title" output.check new.block
  format.depoimento new.block
  format.journal output
  address "address" bibinfo.check output
  publisher "publisher" bibinfo.check output
  format.volume output
  format.number output
  format.pages output
  format.date output new.block
  format.numpages output new.block
  format.doi output new.block
  format.note output new.block
  fin.entry
}

% ENTRADAS: address,depoimento,journal,journal-sub,author,editor,compilador,organization,entidade,entidade-sub,year,month,day,title,subtitle,pages,chapter, doi,note
FUNCTION {newspaper}
{ output.bibitem
  formatar.autor.ou.organizador.ou.editor.ou.compilador new.sentence
  format.title "title" output.check new.block
  format.depoimento new.block
  format.journal output
  address "address" bibinfo.check output
  format.full.date new.block
  format.bchapter.pages new.sentence
  format.doi output new.block
  format.note output new.block
  fin.entry
}

% ENTRADAS: author,entidade,entidade-sub,title,subtitle,year,month,day,number,doi,note
FUNCTION {patent}
{ output.bibitem
  formatar.entidade.e.ou.autor new.sentence
  format.btitle output new.sentence
  format.patent.number
  format.full.date new.block
  format.doi output new.block
  format.note output new.block
  fin.entry
}

% ENTRADAS: address,jurisdiction,booktitle,booksubtitle,year,month,day,title,subtitle,pages,volume,number,doi,note
FUNCTION {juridical}
{ output.bibitem
  formatar.jurisdicao new.sentence
  format.title "title" output.check new.block
  format.booktitle output new.block
  address "address" bibinfo.check output
  format.volume output
  format.number output
  format.pages output
  format.full.date new.block
  format.doi output new.block
  format.note output new.block
  fin.entry
}

% ENTRADAS: note,forma
FUNCTION {other}
{ output.bibitem
  format.note output
  fin.entry
}

FUNCTION {default.type} { other }
READ
FUNCTION {sortify}
{ purify$
  "l" change.case$
}
INTEGERS { len }
FUNCTION {chop.word}
{ 's :=
  'len :=
  s #1 len substring$ =
    { s len #1 + global.max$ substring$ }
    's
  if$
}
FUNCTION {sort.format.names}
{ 's :=
  #1 'nameptr :=
  ""
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { format.single.name 't :=
      nameptr #1 >
        {
          nameptr #3
          #1 + =
          numnames #3
          > and
            { "others" 't :=
              #1 'namesleft := }
            'skip$
          if$
          "   "  *
          namesleft #1 = t "others" = and
            { "zzzzz" * }
            { t sortify * }
          if$
        }
        { t sortify * }
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}

FUNCTION {sort.format.title}
{ 't :=
  "A " #2
    "An " #3
      "The " #4 t chop.word
    chop.word
  chop.word
  sortify
  #1 global.max$ substring$
}

FUNCTION {patent.sort}
{ entidade empty$ 'skip$ {#1 'escolhido := entidade sortify} if$ }

FUNCTION {any.sort}
{ #0 'escolhido :=
	type$ "patent" = {patent.sort} 'skip$ if$
	jurisdiction empty$ escolhido or 'skip$ {#1 'escolhido := jurisdiction sortify} if$
	author empty$ escolhido or 'skip$ {#1 'escolhido := author sort.format.names} if$
	editor empty$ escolhido or 'skip$ {#1 'escolhido := editor sort.format.names} if$
	compilador empty$ escolhido or 'skip$ {#1 'escolhido := compilador sort.format.names} if$
	organization empty$ escolhido or 'skip$ {#1 'escolhido := organization sort.format.names} if$
	entidade empty$ escolhido or 'skip$ {#1 'escolhido := entidade sortify} if$
	title empty$ escolhido or 'skip$ {#1 'escolhido := title sortify} if$
	journal empty$ escolhido or 'skip$ {#1 'escolhido := journal sortify} if$
	evento empty$ escolhido or 'skip$ {#1 'escolhido := evento sortify} if$
	escolhido 'skip$ {note empty$ { "precisa de campos in " cite$ * warning$ ""} {note sortify} if$} if$
}

FUNCTION {presort}
{ 
 any.sort
  "    " * year field.or.null sortify * "    " * title field.or.null
  sort.format.title * #1 entry.max$ substring$ 'sort.key$ :=
  %for non-alpha style we drop a void sort.key such that SORT has no effect
  year empty$ {""} {year} if$ 'year.label :=
  calc.extended.label
}

FUNCTION {set.default.variables}
{  "\altdestaque" 'enfase :=
  #3 'usp.max.nomes :=
}
EXECUTE {set.default.variables}
ITERATE {presort}
SORT
STRINGS { last.label next.extra}
INTEGERS { last.extra.num }
FUNCTION {forward.pass}
{ last.label label =
    { last.extra.num #1 + 'last.extra.num :=
      last.extra.num int.to.chr$ 'extra.label :=
    }
    { "a" chr.to.int$ 'last.extra.num :=
      "" 'extra.label :=
      label 'last.label :=
    }
  if$
}

FUNCTION {reverse.pass}
{ next.extra "b" =
    { "a" 'extra.label := }
    'skip$
  if$
  label extra.label * 'label :=
  year.label extra.label * 'year.label :=
  extra.label 'next.extra :=
  label "}" * 'label :=
}
ITERATE {forward.pass}
REVERSE {reverse.pass}
FUNCTION {begin.bib}
{
  "\providecommand{\adicionaformascitacoes}[4]{}" write$ newline$
  "\providecommand{\uspbibitem}{\bibitem}" write$ newline$
  "\providecommand{\altdestaque}[1]{\textbf{#1}}" write$ newline$
  "\providecommand{\usareferenciaalfabetica}{}" write$ newline$
  "\providecommand{\href}[2]{#2}" write$ newline$
  "\usareferenciaalfabetica" write$ newline$  
  "\begin{thebibliography}{}" write$ newline$
  preamble$ empty$
    'skip$
    { preamble$ write$ newline$ }
  if$
}
EXECUTE {begin.bib}
EXECUTE {init.state.consts}
FUNCTION {process.entry}{call.type$}
ITERATE {process.entry}
FUNCTION {end.bib}
{ newline$ "\end{thebibliography}" write$ newline$
}
EXECUTE {end.bib}
