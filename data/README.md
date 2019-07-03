# raw

## gtr_projects.csv

Gateway to Research (https://gtr.ukri.org/) projects dataset on publicly funded
research and innovation. The projects in the dataset represent those that were
successfully funded and have been completed or are in the process of being
carried out. The project descriptions were written at the time of proposal.

Origin: Tier 0

[Data profile](../reports/eda/gtr_projects_profile.html)

### Columns

`id (str)`: Unique GtR ID for the project (URL)
`end (date)`: End date of the project
`title (str)`: Title of the project
`status (str)`: Active status of the project (categorical), e.g. 'Closed'
`grantCategory (str)`: Type of funding award (categorical), e.g. 'Studentship'
`leadFunder (str)`: Who mainly funded the research (categorical)
`abstractText (str)`: Long text describing the research project
`start (date)`: Start date of the project
`created (datetime)`: Date and time of data collection
`leadOrganisationDepartment (str)`: Name of the department within the organisation recieving the funding
`potentialImpact (str)`: Long text detailing the possible impact of the research
`techAbstractText (str)`: Long text describing technical details of the research project
