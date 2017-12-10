import { RouterModule, Routes } from '@angular/router';
import { TestComponent } from './components/pages/test/test.component';
import { TimelineComponent } from './components/pages/timeline/timeline.component';

export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'home'},
  { path: 'home', component: TestComponent },
  { path: 'timeline', component: TimelineComponent }
];

export const APP_ROUTES = RouterModule.forRoot(routes);
